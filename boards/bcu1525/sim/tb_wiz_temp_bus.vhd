-- docs/bcu1525_sysmon.md: the System Management Wizard's temperature bus FSM (port A) and its
-- DRP arbiter, as a build generates them (system_management_wiz v1.3 rev 25; AMD's files, not in
-- this repository: a WG build's OUT_DIR/otpu.gen has them), with the AXI path's one-cycle DRP
-- requests (port B) and a model DRP (DRDY 4 cycles after DEN). For each K from 0 to 1010 one B
-- request lands K cycles after the FSM raised busy (K = 0: busy's first cycle, before its den),
-- then the die changes: a trial whose temp_out keeps the old code is frozen. B's next request
-- is checked to complete. Expected: "the bus FSM's period: 994 cycles", "K=0 frozen",
-- "trials 1011, frozen 1".
--
--   S=OUT_DIR/otpu.gen/sources_1/bd/otpu_bdw/ip/otpu_bdw_sysmon_0
--   ghdl -a --std=08 -fsynopsys $S/otpu_bdw_sysmon_0_temperature_update.vhd \
--       $S/otpu_bdw_sysmon_0_drp_arbiter.vhd tb_wiz_temp_bus.vhd
--   ghdl -e --std=08 -fsynopsys tb_wiz_temp_bus && ./tb_wiz_temp_bus
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

entity tb_wiz_temp_bus is
end tb_wiz_temp_bus;

architecture sim of tb_wiz_temp_bus is
  signal clk : std_logic := '0';
  signal rst : std_logic := '1';
  signal daddr_A, daddr_C : std_logic_vector(7 downto 0);
  signal den_A, dwe_A, drdy_A, bbusy_A : std_logic;
  signal di_A, do_A, di_C, do_C, do_B : std_logic_vector(15 downto 0);
  signal den_B : std_logic := '0';
  signal drdy_B, den_C, dwe_C, drdy_C : std_logic;
  signal temp_out : std_logic_vector(9 downto 0);
  signal wait_cycle : std_logic_vector(15 downto 0) := x"03E8";
  signal busy_q : std_logic := '0';
  signal since : integer := 0;              -- cycles since bbusy_A rose (0 its first cycle high)
  signal arm : boolean := false;
  signal k_now : integer := 0;
  signal fired : boolean := false;
  signal poke : std_logic := '0';
  signal b_done : integer := 0;
  signal die : unsigned(9 downto 0) := to_unsigned(600, 10);
  -- the DRP: DRDY 4 cycles after DEN, DO the die's code in [15:6]
  type pipe_t is array (0 to 3) of std_logic;
  signal pipe : pipe_t := (others => '0');
begin
  clk <= not clk after 5 ns;

  u_t : entity work.temperature_update port map (
    reset => rst, clk => clk, temp_bus_update => '0', wait_cycle => wait_cycle,
    temp_out => temp_out, daddr_o => daddr_A, den_o => den_A, di_o => di_A, dwe_o => dwe_A,
    do_i => do_A, drdy_i => drdy_A, busy_o => bbusy_A);
  u_a : entity work.drp_arbiter port map (
    reset => rst, clk => clk, jtaglocked => '0', bgrant_A => open, bgrant_B => open,
    bbusy_A => bbusy_A, bbusy_B => '0', daddr_A => daddr_A, den_A => den_A, di_A => di_A,
    dwe_A => dwe_A, do_A => do_A, drdy_A => drdy_A, daddr_B => x"00", den_B => den_B,
    di_B => x"0000", dwe_B => '0', do_B => do_B, drdy_B => drdy_B, daddr_C => daddr_C,
    den_C => den_C, di_C => di_C, dwe_C => dwe_C, do_C => do_C, drdy_C => drdy_C);

  drp : process (clk)
  begin
    if rising_edge(clk) then
      pipe <= den_C & pipe(0 to 2);
    end if;
  end process;
  drdy_C <= pipe(3);
  do_C <= std_logic_vector(die) & "000000";

  -- B's request: one cycle, `since` = K, once per trial (as den_actual: den_i and not den_d1)
  -- K = 0: the FSM's first busy cycle (busy high, den not yet); K > 0: `since` = K (1 the cycle of
  -- its den, period - 1 the cycle before the next busy)
  den_B <= '1' when poke = '1' or (arm and not fired and
                    ((k_now = 0 and bbusy_A = '1' and busy_q = '0') or
                     (k_now > 0 and since = k_now and not (bbusy_A = '1' and busy_q = '0'))))
           else '0';

  mon : process (clk)
  begin
    if rising_edge(clk) then
      busy_q <= bbusy_A;
      if bbusy_A = '1' and busy_q = '0' then
        since <= 1;
      else
        since <= since + 1;
      end if;
      if rst = '1' then
        fired <= false;
      elsif den_B = '1' then
        fired <= true;
      end if;
      if drdy_B = '1' then
        b_done <= b_done + 1;
      end if;
    end if;
  end process;

  stim : process
    variable l : line;
    variable stuck, frozen_n, b_ok : integer;
    variable b0 : integer;
    variable t0 : std_logic_vector(9 downto 0);
  begin
    frozen_n := 0;
    rst <= '0';
    wait until rising_edge(clk) and bbusy_A = '1' and busy_q = '0';
    stuck := 0;
    loop
      wait until rising_edge(clk);
      stuck := stuck + 1;
      exit when bbusy_A = '1' and busy_q = '0';
    end loop;
    write(l, string'("the bus FSM's period: ") & integer'image(stuck) & string'(" cycles"));
    writeline(output, l);
    for k in 0 to 1010 loop
      -- (since: 0 is never seen with bbusy_A high: it is 1 in busy's first cycle; K = 1 that cycle)
      rst <= '1';
      arm <= false;
      wait until rising_edge(clk);
      wait until rising_edge(clk);
      rst <= '0';
      die <= to_unsigned(600, 10);
      for i in 0 to 2100 loop                -- two of the bus's periods
        wait until rising_edge(clk);
      end loop;
      k_now <= k;
      wait until rising_edge(clk);
      arm <= true;
      wait until rising_edge(clk) and bbusy_A = '1' and busy_q = '0';   -- a period starts
      for i in 0 to 1100 loop
        wait until rising_edge(clk);
      end loop;
      arm <= false;
      die <= to_unsigned(700, 10);          -- the die moves; a working bus follows
      for i in 0 to 2100 loop
        wait until rising_edge(clk);
      end loop;
      -- B after it: one more request must still complete
      b0 := b_done;
      poke <= '1';
      wait until rising_edge(clk);
      poke <= '0';
      for i in 0 to 40 loop
        wait until rising_edge(clk);
      end loop;
      if b_done /= b0 + 1 then
        write(l, string'("K=") & integer'image(k) & string'(" B's request lost"));
        writeline(output, l);
      end if;
      if temp_out /= std_logic_vector(to_unsigned(700, 10)) then
        frozen_n := frozen_n + 1;
        write(l, string'("K=") & integer'image(k) & string'(" frozen: temp_out ")
              & integer'image(to_integer(unsigned(temp_out))) & string'(" busy ")
              & std_logic'image(bbusy_A) & string'(" B done ") & integer'image(b_done));
        writeline(output, l);
      end if;
    end loop;
    write(l, string'("trials 1011, frozen ") & integer'image(frozen_n));
    writeline(output, l);
    std.env.finish;
  end process;
end sim;
