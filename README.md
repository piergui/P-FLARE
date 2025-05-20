# P-FLARE
Penalised FLux-driven Algorithm for REduced models

## Application to the Flux-Driven Hasegawa-Wakatani (FDHW) system

Below are some animations of numerical experiments which can be done with the code applied to the FDHW

### Spreading of a profile with a large slope on the left
We take an initial profile which is very steep on the left, and completely flat on the right.
<img src="https://github.com/piergui/fd_hwak/blob/main/outfdC0.05_32pi_1024x1024_kapl5.0.gif"/>
- The linear instability occurs mostly where the profile is the steepest, which is where turbulence develops.
- With, the profile relaxes due to the outward particle flux, and the turbulent region slowly spreads to the right until it reaches the right edge  .
- The final state is dominated by zonal flows: since $\kappa$ decreases due to the turbulent particle flux, we reach $C/\kappa \sim 0.1$ and the system transitions to a zonal flows dominated state. Then zonal flows decrease the transport and the system is "frozen" by the zonal flows.

Below we can look at the time evolution of the density and zonal velocity radial profiles for a similar simulation with a padded resolution of 4096 $\times$ 4096
<img src="https://github.com/piergui/fd_hwak/blob/main/outfdC0.05_64pi_4096x4096_kapl5.0_profiles.gif"/>

#### Comparison with a 1D reduced model for profile relaxation and turbulence spreading
<img src="https://github.com/piergui/fd_hwak/blob/main/DNS_vs_1D.gif"/>

### Adding a localised particle source
<img src="https://github.com/piergui/fd_hwak/blob/main/4_sources.gif" width="100%" height="100%"/>
