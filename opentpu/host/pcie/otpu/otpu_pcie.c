// SPDX-License-Identifier: GPL-2.0
/*
 * otpu_pcie: the openTPU card's driver for its LitePCIe bitstreams (PCI 10ee:5054, subsystem
 * 10ee:4c50; the XDMA bitstreams, 10ee:7028, stay with Xilinx's xdma driver). docs/litepcie.md.
 *
 * Per card N, three nodes with the XDMA driver's names and semantics, so the host's transport
 * (opentpu/host/board.py) drives either:
 *   /dev/otpuN_user    mmap: BAR0's user window (the first OTLP_USER_SIZE bytes: the accelerator's
 *                      control registers, the LiteDRAM CSRs, the XADC). LitePCIe's own CSRs above
 *                      it program DMA engines that can write anywhere in host memory: no mapping.
 *   /dev/otpuN_h2c_0   pwrite(fd, buf, n, card): DMA host -> card of n bytes to the card byte
 *                      address card; returns once the card's memory holds them.
 *   /dev/otpuN_c2h_0   pread(fd, buf, n, card): DMA card -> host.
 * buf, n and card are multiples of 16 bytes (the DMA's beat; EINVAL otherwise), card + n at most
 * 4 GiB (bit 31: the DDR3 channel).
 *
 * A transfer pins the caller's pages (no copy) and goes to the card in commands of up to CHUNK
 * bytes, up to INFLIGHT of them queued per direction: each is one command in otpu_dma's queue (the
 * card address and length, which otpu_pcie_mm turns into memory bursts) and the scatter-gather
 * descriptors of its pages in LitePCIe's DMA table (the reader's for H2C, the writer's for C2H;
 * none crosses a 4 KiB boundary, except from a 4 KiB-aligned start: the DMA splits them into
 * reads of the max read request and writes of the max payload, which must not cross one). The
 * card counts finished commands (h2c_count: the last write response is in; c2h_count: the writer
 * has sent the command's last descriptor) and writes both counts to a record in host memory on
 * every change, behind the transfer's own writes, then raises an MSI. A waiter spins on the record
 * for spin_us, then sleeps until an MSI, after which it spins up to msi_spin_us for a record still
 * on its way (sysfs msi_early counts those; the card delays the MSI behind the record's write), and
 * looks again every 200 us whatever happens (timer_wakes). A command's pages are released once its
 * count is reached.
 *
 * On a timeout the DMA is reset (LitePCIe's reader and writer, their tables, otpu_pcie_mm and the
 * memory port's host side) and the call fails with EIO.
 */
#include <linux/module.h>
#include <linux/version.h>
#include <linux/pci.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/interrupt.h>
#include <linux/wait.h>
#include <linux/ktime.h>
#include <linux/delay.h>
#include <linux/scatterlist.h>
#include <linux/dma-mapping.h>
#include <linux/idr.h>

#include "csr.h"

#define DRV_NAME	"otpu_pcie"
#define OTPU_NODES	3		/* user, h2c_0, c2h_0 */
#define OTPU_MAX_CARDS	16
#define CHUNK		(256u << 10)	/* bytes a command (65 descriptors at most: 4 in a table) */
#define INFLIGHT	4		/* commands queued per direction */
#define CHUNK_PAGES	(CHUNK / PAGE_SIZE + 1)
#define DESC_MAX	(8u << 20)	/* bytes a descriptor (LitePCIe: 24-bit lengths) */
#define DESC_IRQ_DIS	(1u << 24)	/* the descriptor's upper word: irq_disable */
#define DESC_LAST_DIS	(1u << 25)	/* last_disable */
#define MSI_REC		(1u << 0)	/* a completion record written */
#define MSI_ERR		(1u << 1)	/* otpu_pcie_mm's error */

static unsigned int spin_us = 20;
module_param(spin_us, uint, 0644);
MODULE_PARM_DESC(spin_us, "microseconds a transfer's waiter spins on the completion record before it sleeps until the MSI (default 20)");
static unsigned int msi_spin_us = 2;
module_param(msi_spin_us, uint, 0644);
MODULE_PARM_DESC(msi_spin_us, "microseconds a waiter woken by an MSI spins for a record still on its way (default 2)");
static unsigned int timeout_ms = 10000;
module_param(timeout_ms, uint, 0644);
MODULE_PARM_DESC(timeout_ms, "a command's longest wait before the DMA is reset (default 10000)");

/* the completion record the card writes (otpu_dma: 16 bytes) */
struct otpu_rec {
	u32 h2c;
	u32 c2h;
	u32 mm_status;
	u32 seq;
};

struct otpu_cmd {
	struct page **pages;
	int npages;
	struct sg_table sgt;
	bool mapped;
	int ndesc;
	u32 target;			/* the count that finishes it */
};

struct otpu_dir {
	struct mutex lock;
	bool h2c;
	u32 issued;			/* commands given to the card (mod 2^32) */
	struct otpu_cmd cmd[INFLIGHT];
	int head, n;			/* the queued commands, oldest first */
	int table;			/* descriptors they hold in the table */
	u32 cmd_addr, cmd_len;		/* otpu_dma's command CSRs */
	u32 value, we;			/* LitePCIe's table CSRs */
};

struct otpu_dev {
	struct pci_dev *pdev;
	void __iomem *bar;
	resource_size_t bar_start;
	resource_size_t bar_len;
	int idx;
	struct cdev cdev;
	struct device *node[OTPU_NODES];
	struct otpu_rec *rec;
	dma_addr_t rec_dma;
	wait_queue_head_t wq;
	struct mutex reset_lock;
	struct otpu_dir h2c, c2h;
	atomic_t irqs;
	atomic_t resets;
	atomic_t msi_early;		/* MSIs that came before their record */
	atomic_t timer_wakes;		/* records found by the 200 us timer, not an MSI */
};

static const char *const node_name[OTPU_NODES] = { "user", "h2c_0", "c2h_0" };
static struct class *otpu_class;
static dev_t otpu_devt;
static DEFINE_IDA(otpu_ida);

static inline u32 rd(struct otpu_dev *d, u32 off) { return ioread32(d->bar + off); }
static inline void wr(struct otpu_dev *d, u32 off, u32 v) { iowrite32(v, d->bar + off); }

/* the DMA from scratch: engines off, tables empty and in program mode, otpu_pcie_mm and the
 * memory port's host side reset, the counts taken as they stand, the record on, engines on */
static void otpu_dma_init(struct otpu_dev *d)
{
	wr(d, CSR_PCIE_DMA0_READER_ENABLE_ADDR, 0);
	wr(d, CSR_PCIE_DMA0_WRITER_ENABLE_ADDR, 0);
	wr(d, CSR_OTPU_DMA_WB_CTRL_ADDR, 0);
	wr(d, CSR_OTPU_DMA_MM_CTRL_ADDR, 1);
	wr(d, CSR_PCIE_DMA0_READER_TABLE_LOOP_PROG_N_ADDR, 0);
	wr(d, CSR_PCIE_DMA0_WRITER_TABLE_LOOP_PROG_N_ADDR, 0);
	wr(d, CSR_PCIE_DMA0_READER_TABLE_RESET_ADDR, 1);
	wr(d, CSR_PCIE_DMA0_WRITER_TABLE_RESET_ADDR, 1);
	(void)rd(d, CSR_OTPU_DMA_ID_ADDR);	/* the writes are in */
	udelay(20);				/* the memory port's resets (otpu_mem_ch: tens of cycles) */
	wr(d, CSR_OTPU_DMA_MM_CTRL_ADDR, 0);
	d->h2c.issued = rd(d, CSR_OTPU_DMA_H2C_COUNT_ADDR);
	d->c2h.issued = rd(d, CSR_OTPU_DMA_C2H_COUNT_ADDR);
	WRITE_ONCE(d->rec->h2c, d->h2c.issued);
	WRITE_ONCE(d->rec->c2h, d->c2h.issued);
	wr(d, CSR_OTPU_DMA_WB_LO_ADDR, lower_32_bits(d->rec_dma));
	wr(d, CSR_OTPU_DMA_WB_HI_ADDR, upper_32_bits(d->rec_dma));
	wr(d, CSR_OTPU_DMA_WB_CTRL_ADDR, 1);
	wr(d, CSR_PCIE_DMA0_READER_ENABLE_ADDR, 1);
	wr(d, CSR_PCIE_DMA0_WRITER_ENABLE_ADDR, 1);
	(void)rd(d, CSR_OTPU_DMA_ID_ADDR);
}

static void otpu_dma_off(struct otpu_dev *d)
{
	wr(d, CSR_PCIE_DMA0_READER_ENABLE_ADDR, 0);
	wr(d, CSR_PCIE_DMA0_WRITER_ENABLE_ADDR, 0);
	wr(d, CSR_OTPU_DMA_WB_CTRL_ADDR, 0);
	wr(d, CSR_PCIE_MSI_ENABLE_ADDR, 0);
	(void)rd(d, CSR_OTPU_DMA_ID_ADDR);
}

/* one descriptor into a table: its upper word (length, controls), the address's lower word
 * (which commits the 64-bit value), then the upper address word, whose write queues it */
static void push_desc(struct otpu_dev *d, struct otpu_dir *q, u64 addr, u32 len, bool irq)
{
	wr(d, q->value, len | (irq ? 0 : DESC_IRQ_DIS) | DESC_LAST_DIS);
	wr(d, q->value + 4, lower_32_bits(addr));
	wr(d, q->we, upper_32_bits(addr));
}

static void cmd_release(struct otpu_dev *d, struct otpu_dir *q, struct otpu_cmd *c)
{
	if (c->mapped)
		dma_unmap_sgtable(&d->pdev->dev, &c->sgt, q->h2c ? DMA_TO_DEVICE : DMA_FROM_DEVICE, 0);
	c->mapped = false;
	sg_free_table(&c->sgt);
	if (q->h2c)
		unpin_user_pages(c->pages, c->npages);
	else
		unpin_user_pages_dirty_lock(c->pages, c->npages, true);
	c->npages = 0;
	q->table -= c->ndesc;
	c->ndesc = 0;
}

/* the descriptors of one mapped command, in order; the C2H one's last enables the writer's IRQ
 * (its count). Returns how many */
static int cmd_descs(struct otpu_dev *d, struct otpu_dir *q, struct otpu_cmd *c)
{
	struct scatterlist *sg;
	u64 pa = 0;
	u32 pl = 0;
	int i, n = 0;

	for_each_sgtable_dma_sg(&c->sgt, sg, i) {
		u64 a = sg_dma_address(sg);
		u32 l = sg_dma_len(sg);

		while (l) {
			u32 k = (a & 4095) ? min_t(u32, l, 4096 - (a & 4095)) : min_t(u32, l, DESC_MAX);

			if (pl)
				push_desc(d, q, pa, pl, false);
			pa = a;
			pl = k;
			n++;
			a += k;
			l -= k;
		}
	}
	if (pl)
		push_desc(d, q, pa, pl, !q->h2c);
	return n;
}

static int cmd_issue(struct otpu_dev *d, struct otpu_dir *q, unsigned long ubuf, u32 n, u32 card)
{
	struct otpu_cmd *c = &q->cmd[(q->head + q->n) % INFLIGHT];
	unsigned long first = ubuf >> PAGE_SHIFT, last = (ubuf + n - 1) >> PAGE_SHIFT;
	int np = last - first + 1, got, ret;

	got = pin_user_pages_fast(ubuf & PAGE_MASK, np, q->h2c ? 0 : FOLL_WRITE, c->pages);
	if (got != np) {
		if (got > 0)
			unpin_user_pages(c->pages, got);
		return got < 0 ? got : -EFAULT;
	}
	c->npages = np;
	ret = sg_alloc_table_from_pages_segment(&c->sgt, c->pages, np, offset_in_page(ubuf), n,
						DESC_MAX, GFP_KERNEL);
	if (ret)
		goto unpin;
	ret = dma_map_sgtable(&d->pdev->dev, &c->sgt, q->h2c ? DMA_TO_DEVICE : DMA_FROM_DEVICE, 0);
	if (ret)
		goto free;
	c->mapped = true;
	if (q->h2c) {
		/* the command, then the data it takes from the reader's stream */
		wr(d, q->cmd_addr, card);
		wr(d, q->cmd_len, n);
		c->ndesc = cmd_descs(d, q, c);
	} else {
		/* the writer's descriptors first, so the stream finds them */
		c->ndesc = cmd_descs(d, q, c);
		wr(d, q->cmd_addr, card);
		wr(d, q->cmd_len, n);
	}
	q->table += c->ndesc;
	c->target = ++q->issued;
	q->n++;
	return 0;
free:
	sg_free_table(&c->sgt);
unpin:
	unpin_user_pages(c->pages, np);
	c->npages = 0;
	return ret;
}

static bool rec_reached(struct otpu_dev *d, struct otpu_dir *q, u32 target)
{
	u32 cnt = q->h2c ? READ_ONCE(d->rec->h2c) : READ_ONCE(d->rec->c2h);

	return (s32)(cnt - target) >= 0;
}

static int cmd_wait(struct otpu_dev *d, struct otpu_dir *q, u32 target)
{
	ktime_t t0 = ktime_get();
	ktime_t end = ktime_add_ms(t0, timeout_ms);
	int seen;

	while (ktime_us_delta(ktime_get(), t0) < spin_us) {
		if (rec_reached(d, q, target))
			goto done;
		cpu_relax();
	}
	for (;;) {
		seen = atomic_read(&d->irqs);
		if (rec_reached(d, q, target))
			break;
		/* woken by any MSI (either direction's, or an error), or every 200 us */
		if (!wait_event_hrtimeout(d->wq, rec_reached(d, q, target) || atomic_read(&d->irqs) != seen,
					  us_to_ktime(200)) && rec_reached(d, q, target))
			break;
		if (atomic_read(&d->irqs) != seen) {
			/* the card writes the record, then raises the MSI (MSI_DELAY cycles later), but
			 * the two reach host memory by different paths: give a record behind its MSI a
			 * moment */
			ktime_t t1 = ktime_get();

			while (ktime_us_delta(ktime_get(), t1) < msi_spin_us) {
				if (rec_reached(d, q, target)) {
					atomic_inc(&d->msi_early);
					goto done;
				}
				cpu_relax();
			}
		} else if (rec_reached(d, q, target)) {
			atomic_inc(&d->timer_wakes);
			break;
		}
		if (ktime_after(ktime_get(), end))
			return rec_reached(d, q, target) ? 0 : -ETIMEDOUT;
	}
done:
	dma_rmb();	/* the record before the data it vouches for */
	return 0;
}

static void otpu_recover(struct otpu_dev *d, struct otpu_dir *q, int err)
{
	dev_err(&d->pdev->dev,
		"%s timed out (%d): issued %u, record %u/%u seq %u, counts %u/%u, queue %08x, mm_status %08x, reader table %u, writer table %u, link %08x, irqs %d; resetting the DMA\n",
		q->h2c ? "H2C" : "C2H", err, q->issued, READ_ONCE(d->rec->h2c), READ_ONCE(d->rec->c2h),
		READ_ONCE(d->rec->seq), rd(d, CSR_OTPU_DMA_H2C_COUNT_ADDR),
		rd(d, CSR_OTPU_DMA_C2H_COUNT_ADDR), rd(d, CSR_OTPU_DMA_QUEUE_ADDR),
		rd(d, CSR_OTPU_DMA_MM_STATUS_ADDR), rd(d, CSR_PCIE_DMA0_READER_TABLE_LEVEL_ADDR),
		rd(d, CSR_PCIE_DMA0_WRITER_TABLE_LEVEL_ADDR), rd(d, CSR_PCIE_PHY_PHY_LINK_STATUS_ADDR),
		atomic_read(&d->irqs));
	mutex_lock(&d->reset_lock);
	otpu_dma_off(d);
	msleep(2);	/* writes already issued land before the pages go back */
	otpu_dma_init(d);
	wr(d, CSR_PCIE_MSI_ENABLE_ADDR, MSI_REC | MSI_ERR);
	atomic_inc(&d->resets);
	mutex_unlock(&d->reset_lock);
}

static ssize_t otpu_dma(struct otpu_dev *d, struct otpu_dir *q, unsigned long ubuf, size_t len,
			loff_t card)
{
	size_t done = 0;
	int ret = 0;

	if (!len)
		return 0;
	if ((ubuf | len | card) & 15 || card < 0 || card + len > (1ull << 32))
		return -EINVAL;
	if (mutex_lock_interruptible(&q->lock))
		return -ERESTARTSYS;
	while (done < len || q->n) {
		if (done < len && q->n < INFLIGHT) {
			u32 n = min_t(size_t, len - done, CHUNK);
			int worst = ((ubuf + done + n - 1) >> PAGE_SHIFT) - ((ubuf + done) >> PAGE_SHIFT) + 2;

			if (q->table + worst <= OTLP_TABLE_DEPTH) {
				ret = cmd_issue(d, q, ubuf + done, n, (u32)(card + done));
				if (ret)
					goto fail;
				done += n;
				continue;
			}
		}
		ret = cmd_wait(d, q, q->cmd[q->head].target);
		if (ret)
			goto fail;
		cmd_release(d, q, &q->cmd[q->head]);
		q->head = (q->head + 1) % INFLIGHT;
		q->n--;
	}
	mutex_unlock(&q->lock);
	return len;
fail:
	if (q->n)
		otpu_recover(d, q, ret);
	while (q->n) {
		cmd_release(d, q, &q->cmd[q->head]);
		q->head = (q->head + 1) % INFLIGHT;
		q->n--;
	}
	mutex_unlock(&q->lock);
	return ret == -ETIMEDOUT ? -EIO : ret;
}

static int otpu_open(struct inode *inode, struct file *f)
{
	struct otpu_dev *d = container_of(inode->i_cdev, struct otpu_dev, cdev);

	f->private_data = d;
	return 0;
}

static int node_of(struct file *f)
{
	return iminor(file_inode(f)) % OTPU_NODES;
}

static ssize_t otpu_write(struct file *f, const char __user *buf, size_t len, loff_t *pos)
{
	struct otpu_dev *d = f->private_data;
	ssize_t r;

	if (node_of(f) != 1)
		return -EINVAL;
	r = otpu_dma(d, &d->h2c, (unsigned long)buf, len, *pos);
	if (r > 0)
		*pos += r;
	return r;
}

static ssize_t otpu_read(struct file *f, char __user *buf, size_t len, loff_t *pos)
{
	struct otpu_dev *d = f->private_data;
	ssize_t r;

	if (node_of(f) != 2)
		return -EINVAL;
	r = otpu_dma(d, &d->c2h, (unsigned long)buf, len, *pos);
	if (r > 0)
		*pos += r;
	return r;
}

static int otpu_mmap(struct file *f, struct vm_area_struct *vma)
{
	struct otpu_dev *d = f->private_data;
	unsigned long size = vma->vm_end - vma->vm_start;

	if (node_of(f) != 0 || vma->vm_pgoff != 0 || size > OTLP_USER_SIZE || size > d->bar_len)
		return -EINVAL;
	vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
	vm_flags_set(vma, VM_IO | VM_DONTEXPAND | VM_DONTDUMP);
	return io_remap_pfn_range(vma, vma->vm_start, d->bar_start >> PAGE_SHIFT, size,
				  vma->vm_page_prot);
}

static const struct file_operations otpu_fops = {
	.owner = THIS_MODULE,
	.open = otpu_open,
	.read = otpu_read,
	.write = otpu_write,
	.mmap = otpu_mmap,
	.llseek = default_llseek,
};

static irqreturn_t otpu_isr(int irq, void *p)
{
	struct otpu_dev *d = p;

	wr(d, CSR_PCIE_MSI_CLEAR_ADDR, MSI_REC | MSI_ERR);
	atomic_inc(&d->irqs);
	wake_up_all(&d->wq);
	return IRQ_HANDLED;
}

/* sysfs, on the user node: the DMA's state for otpu-setup --check and diagnostics */
#define OTPU_ATTR_RD(name, expr)						\
static ssize_t name##_show(struct device *dev, struct device_attribute *a, char *buf) \
{									\
	struct otpu_dev *d = dev_get_drvdata(dev);			\
	return sysfs_emit(buf, "%u\n", (unsigned int)(expr));		\
}									\
static DEVICE_ATTR_RO(name)
OTPU_ATTR_RD(h2c_count, rd(d, CSR_OTPU_DMA_H2C_COUNT_ADDR));
OTPU_ATTR_RD(c2h_count, rd(d, CSR_OTPU_DMA_C2H_COUNT_ADDR));
OTPU_ATTR_RD(mm_status, rd(d, CSR_OTPU_DMA_MM_STATUS_ADDR));
OTPU_ATTR_RD(link_status, rd(d, CSR_PCIE_PHY_PHY_LINK_STATUS_ADDR));
OTPU_ATTR_RD(max_payload, rd(d, CSR_PCIE_PHY_PHY_MAX_PAYLOAD_SIZE_ADDR));
OTPU_ATTR_RD(max_read_request, rd(d, CSR_PCIE_PHY_PHY_MAX_REQUEST_SIZE_ADDR));
OTPU_ATTR_RD(irqs, atomic_read(&d->irqs));
OTPU_ATTR_RD(resets, atomic_read(&d->resets));
OTPU_ATTR_RD(msi_early, atomic_read(&d->msi_early));
OTPU_ATTR_RD(timer_wakes, atomic_read(&d->timer_wakes));
static struct attribute *otpu_attrs[] = {
	&dev_attr_h2c_count.attr, &dev_attr_c2h_count.attr, &dev_attr_mm_status.attr,
	&dev_attr_link_status.attr, &dev_attr_max_payload.attr, &dev_attr_max_read_request.attr,
	&dev_attr_irqs.attr, &dev_attr_resets.attr, &dev_attr_msi_early.attr,
	&dev_attr_timer_wakes.attr, NULL,
};
ATTRIBUTE_GROUPS(otpu);

static void dir_init(struct otpu_dir *q, bool h2c)
{
	mutex_init(&q->lock);
	q->h2c = h2c;
	q->cmd_addr = h2c ? CSR_OTPU_DMA_H2C_ADDR_ADDR : CSR_OTPU_DMA_C2H_ADDR_ADDR;
	q->cmd_len = h2c ? CSR_OTPU_DMA_H2C_LEN_ADDR : CSR_OTPU_DMA_C2H_LEN_ADDR;
	q->value = h2c ? CSR_PCIE_DMA0_READER_TABLE_VALUE_ADDR : CSR_PCIE_DMA0_WRITER_TABLE_VALUE_ADDR;
	q->we = h2c ? CSR_PCIE_DMA0_READER_TABLE_WE_ADDR : CSR_PCIE_DMA0_WRITER_TABLE_WE_ADDR;
}

static int dir_alloc(struct device *dev, struct otpu_dir *q)
{
	int i;

	for (i = 0; i < INFLIGHT; i++) {
		q->cmd[i].pages = devm_kcalloc(dev, CHUNK_PAGES, sizeof(struct page *), GFP_KERNEL);
		if (!q->cmd[i].pages)
			return -ENOMEM;
	}
	return 0;
}

static int otpu_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
	struct otpu_dev *d;
	u32 ident, caps;
	int ret, i;

	d = devm_kzalloc(&pdev->dev, sizeof(*d), GFP_KERNEL);
	if (!d)
		return -ENOMEM;
	d->pdev = pdev;
	init_waitqueue_head(&d->wq);
	mutex_init(&d->reset_lock);
	dir_init(&d->h2c, true);
	dir_init(&d->c2h, false);
	ret = dir_alloc(&pdev->dev, &d->h2c) ?: dir_alloc(&pdev->dev, &d->c2h);
	if (ret)
		return ret;

	ret = pcim_enable_device(pdev);
	if (ret)
		return ret;
	d->bar = pcim_iomap_region(pdev, 0, DRV_NAME);
	if (IS_ERR(d->bar))
		return PTR_ERR(d->bar);
	d->bar_start = pci_resource_start(pdev, 0);
	d->bar_len = pci_resource_len(pdev, 0);
	if (d->bar_len < OTLP_CSR_BASE + 0x10000) {
		dev_err(&pdev->dev, "BAR0 is %llu bytes, not the core's 1 MiB\n", (u64)d->bar_len);
		return -ENODEV;
	}
	ident = rd(d, CSR_OTPU_DMA_ID_ADDR);
	caps = rd(d, CSR_OTPU_DMA_CAPS_ADDR);
	if (ident != OTLP_ID || (caps & 0xff) != OTLP_VERSION || ((caps >> 8) & 0xff) != ilog2(OTLP_TABLE_DEPTH)) {
		dev_err(&pdev->dev, "not an openTPU LitePCIe core this driver knows: id %08x caps %08x\n",
			ident, caps);
		return -ENODEV;
	}
	pci_set_master(pdev);
	ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
	if (ret)
		return ret;
	dma_set_max_seg_size(&pdev->dev, DESC_MAX);	/* an IOMMU may merge pages into one descriptor */
	d->rec = dmam_alloc_coherent(&pdev->dev, PAGE_SIZE, &d->rec_dma, GFP_KERNEL);
	if (!d->rec)
		return -ENOMEM;

	ret = pci_alloc_irq_vectors(pdev, 1, 1, PCI_IRQ_MSI);
	if (ret < 0)
		return ret;
	ret = request_irq(pci_irq_vector(pdev, 0), otpu_isr, 0, DRV_NAME, d);
	if (ret)
		goto free_vec;

	otpu_dma_init(d);
	wr(d, CSR_PCIE_MSI_CLEAR_ADDR, MSI_REC | MSI_ERR);
	wr(d, CSR_PCIE_MSI_ENABLE_ADDR, MSI_REC | MSI_ERR);

	d->idx = ida_alloc_max(&otpu_ida, OTPU_MAX_CARDS - 1, GFP_KERNEL);
	if (d->idx < 0) {
		ret = d->idx;
		goto dma_off;
	}
	cdev_init(&d->cdev, &otpu_fops);
	d->cdev.owner = THIS_MODULE;
	ret = cdev_add(&d->cdev, MKDEV(MAJOR(otpu_devt), d->idx * OTPU_NODES), OTPU_NODES);
	if (ret)
		goto free_idx;
	for (i = 0; i < OTPU_NODES; i++) {
		d->node[i] = device_create_with_groups(otpu_class, &pdev->dev,
			MKDEV(MAJOR(otpu_devt), d->idx * OTPU_NODES + i), d, i == 0 ? otpu_groups : NULL,
			"otpu%d_%s", d->idx, node_name[i]);
		if (IS_ERR(d->node[i])) {
			ret = PTR_ERR(d->node[i]);
			d->node[i] = NULL;
			goto nodes;
		}
	}
	pci_set_drvdata(pdev, d);
	dev_info(&pdev->dev,
		 "otpu%d: link %08x, max payload %u, max read request %u, h2c/c2h counts %u/%u\n",
		 d->idx, rd(d, CSR_PCIE_PHY_PHY_LINK_STATUS_ADDR),
		 rd(d, CSR_PCIE_PHY_PHY_MAX_PAYLOAD_SIZE_ADDR),
		 rd(d, CSR_PCIE_PHY_PHY_MAX_REQUEST_SIZE_ADDR), d->h2c.issued, d->c2h.issued);
	return 0;
nodes:
	for (i = 0; i < OTPU_NODES; i++)
		if (d->node[i])
			device_destroy(otpu_class, MKDEV(MAJOR(otpu_devt), d->idx * OTPU_NODES + i));
	cdev_del(&d->cdev);
free_idx:
	ida_free(&otpu_ida, d->idx);
dma_off:
	otpu_dma_off(d);
	free_irq(pci_irq_vector(pdev, 0), d);
free_vec:
	pci_free_irq_vectors(pdev);
	return ret;
}

static void otpu_remove(struct pci_dev *pdev)
{
	struct otpu_dev *d = pci_get_drvdata(pdev);
	int i;

	for (i = 0; i < OTPU_NODES; i++)
		device_destroy(otpu_class, MKDEV(MAJOR(otpu_devt), d->idx * OTPU_NODES + i));
	cdev_del(&d->cdev);
	otpu_dma_off(d);
	free_irq(pci_irq_vector(pdev, 0), d);
	pci_free_irq_vectors(pdev);
	ida_free(&otpu_ida, d->idx);
}

static const struct pci_device_id otpu_ids[] = {
	{ PCI_DEVICE_SUB(0x10ee, OTLP_DEVICE_ID, 0x10ee, OTLP_SUBSYS_ID) },
	{ 0, }
};
MODULE_DEVICE_TABLE(pci, otpu_ids);

static struct pci_driver otpu_driver = {
	.name = DRV_NAME,
	.id_table = otpu_ids,
	.probe = otpu_probe,
	.remove = otpu_remove,
};

static int __init otpu_init(void)
{
	int ret;

	ret = alloc_chrdev_region(&otpu_devt, 0, OTPU_MAX_CARDS * OTPU_NODES, "otpu");
	if (ret)
		return ret;
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 4, 0)
	otpu_class = class_create("otpu");
#else
	otpu_class = class_create(THIS_MODULE, "otpu");
#endif
	if (IS_ERR(otpu_class)) {
		ret = PTR_ERR(otpu_class);
		goto unreg;
	}
	ret = pci_register_driver(&otpu_driver);
	if (ret)
		goto cls;
	return 0;
cls:
	class_destroy(otpu_class);
unreg:
	unregister_chrdev_region(otpu_devt, OTPU_MAX_CARDS * OTPU_NODES);
	return ret;
}

static void __exit otpu_exit(void)
{
	pci_unregister_driver(&otpu_driver);
	class_destroy(otpu_class);
	unregister_chrdev_region(otpu_devt, OTPU_MAX_CARDS * OTPU_NODES);
}

module_init(otpu_init);
module_exit(otpu_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("openTPU card (LitePCIe bitstreams): BAR0 user window and zero-copy DMA");
MODULE_VERSION("1");
