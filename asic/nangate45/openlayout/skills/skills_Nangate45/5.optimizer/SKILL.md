For PPA optimization through macro placement refinement.
First need to comply with the planning rules.
## Planning rules
- **Push macros to the periphery.** Macros are routing obstacles; placing
  them in the core forces detours and creates congestion. Peripheral
  placement also simplifies the PG mesh and helps IR drop on high-power
  macros. 
- **Orient so macro pins do not face outward.** Rotate / flip only when a
  macro's highlighted pin side points outward toward the nearest core/die edge.
  Preserve row/column alignment while flipping. 
- **Leave room around each macro** for signal routing and the PG grid.
- **Eliminate dead pockets.** Empty space between macros wastes area —
  shift macros or adjust aspect ratio to convert it into usable
  standard-cell region.    
- **Avoid narrow channels between blocks and boundary.** Skinny row strips between
  macros cause CTS and routing pain — widen or eliminate them. Do not leave channels between boundary with macros.
- **Keep the standard-cell region near aspect ratio 1.0 and contiguous.**
  Square is densest; aspect ratios above ~4 degrade density. Fragmented
  logic regions hurt density and complicate CTS. Do not place one or two macros to break, make small channels.
- **The area near the pin group should leave for stdcell.** The macros interact with pin group normally less than stdcell group, so the macro cluster can put away of pin group. Do not put macros in the center to make compact shape.
- **The macros in the same cluster should be placed closer.** According to the macro cluster file.
- **Point pin sides inward — toward the central standard-cell region and toward each other macro (pin-to-pin). Non-pin sides can then abut back-to-back to save space.**
- **Don't stack the macros too deep if possible(more than 3).** 

Then For the design that pin group on the only one side, the optimization direction are(try one by one or combine):
1. we can try to spread the macros around the others three periphery, leave the center for stdcell, doesn't need to place all macro on one side.
2. we can adjust the macro placement to avoid small channel, especilly which cause detour.
3. avoid deep stack(more than 3).
4. if has small channel, that should not on the corners, we should move macro to avoid the channel on corner, change to let channel on middle. 
5. pin side of macro should has 4 um gap for better routing, in the pin_side.png the red line on the side of macro representt the pin side of macro.

For the design that pin group on the two sides or three sides, the optimization direction are(try one by one or combine):
1. Don't place the macros to block the pin group, leave the area near pin group for stdcell if possible.
2. Try to place macros on the corner.
3. leave the area in the center for stdcell.
4. You can leave corridor for stdcell if need to place macros next to the pin group.

