# The physics behind each step

## 1. Three kinds of Sn site

```
   CLOSED                     OPEN                               HYDRATED
   Sn(OSi)4                   (HO)Sn(OSi)3 + HO-Si               Sn(OSi)4(H2O)n

        OSi                        OSi                                OSi   OH2
         |                          |                                  |  /
  SiO -- Sn -- OSi          SiO -- Sn -- OH      HO -- Si       SiO -- Sn -- OSi
         |                          |            (silanol)           /  |
        OSi                        OSi                            H2O   OSi

   4 coordinate               4 coordinate, Sn-O-Si              5 or 6 coordinate
                              bond hydrolysed (+1 H2O)
```

An open site is a closed site plus one water: `Sn-O-Si + H2O -> Sn-OH + HO-Si`.
That is why `E_hydrolysis = E(open) - E(closed) - E(H2O)` is a fair
comparison. In CHA the single T site has four different oxygens, so there are
up to four different open sites, all enumerated by symmetry in step 00.

The builder only guesses geometry. Breaking a Sn-O-Si bond needs the framework
to flex, which a rigid guess cannot do, so step 01 anneals at 400 K and
re-quenches before trusting any energy.

## 2. The collective variable

```
   s = Σ_j f(r_j)        f(r) = (1 - (r/r0)^6) / (1 - (r/r0)^12)

   f
   1 |‾‾‾‾‾\
     |      \             bound water (r ≈ 2.2 Å)   contributes ≈ 0.7 to 0.8
 0.5 |-------●            r = r0                     contributes 0.5
     |        \__
   0 |___________‾‾‾‾‾‾‾  second shell (r > 3.5 Å)   contributes ≈ 0
     0      r0       r
```

r0 is not guessed: step 02 places it at the first minimum of the Sn-O(water)
radial distribution function, where "bound" and "not bound" separate in the
data.

## 3. Well tempered metadynamics in one picture

```
   F(s)                                   V(s,t) grows where the system has been
    |  \           /\                     Hills shrink as V grows:
    |   \   __    /  \                       h = h0 exp(-V(s)/kB ΔT)
    |    \_/  \__/    \___                At long times
    |                                        F(s) = -(γ / (γ - 1)) V(s)
    +--------------------------- s        γ = (T + ΔT) / T, the bias factor
```

The engine is first tested on a tilted double well, where the exact free
energy is known.

## 4. Why the spectrum is weighted by populations

If Sn swaps between 4, 5 and 6 coordination faster than the run, a plain time
average is fine. If it swaps more slowly, one trajectory sees one state and
reports its shift as "the" shift. The equilibrium answer is

```
   <δ> = Σ_k  p_k  <δ>_k        p_k = ∫_basin k exp(-F/kT) ds / Σ
```

with populations from the free energy and state shifts from snapshots taken
inside each basin. This is the first moment of the spectrum. If the states
exchange fast on the NMR time scale the spectrum is one line at <δ>; if they
exchange slowly it is one line per state with weight p_k, and `06` draws that
sum of lines. For the 128 ppm between the 5 and 6 coordinate states (about 19 kHz
at 9.4 T) the exchange rate was not computed.

## 5. Shielding to shift

GIPAW gives absolute shielding σ. Shifts need a reference (SnO2 value from
Aliev, Bartók and Yates, Solid State Nucl. Magn. Reson. 89, 1, 2018):

```
   δ(sample) = σ(SnO2) - σ(sample) + δ(SnO2),     δ(SnO2) = -603.8 ppm vs SnMe4
```

Errors common to both calculations cancel; errors specific to a coordination
change do not. A difference between states removes a constant offset, not an
error that scales with the shift, which was not tested. The most useful output is
still the difference between states (how far hydration moves the shift), and a
short calibration set of Sn(IV) compounds with measured shifts is the next step
before quoting absolute numbers or trusting the size of a difference.

## 6. Why ¹¹⁹Sn is the data poor nucleus

One Sn per cell gives one ¹¹⁹Sn tensor per GIPAW calculation; the same
calculation gives about eleven ²⁹Si and thirty ¹⁷O tensors. `05_gipaw_analysis.py`
prints this count. Any machine learned shift model for Sn therefore needs either
many calculations or a way to borrow information (pre-training on cheaper
labels, then fine tuning on the few Sn labels).
