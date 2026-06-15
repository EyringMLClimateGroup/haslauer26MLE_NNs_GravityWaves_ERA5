# SIMPLIFIED PYTHON VERSION OF LOTT & MILLER (1997) PARAMETERIZATION FOR OROGRAPHIC GRAVITY WAVES
# Elias Haslauer, 2026

import math

import numpy as np


# From SSO_config.f90
    #  ! configuration parameters
    #  ! ------------------------
    #  !
    #  ! threshold values for defining the mask of active points
    #  REAL(wp) :: gpicmea   ! minimum difference "SSO peak height - SSO mean height" [m]
    #  REAL(wp) :: gstd      ! minimum standard deviation of SSO height [m]
    #  !
    #  ! parameters controling the strength of the effects
    #  REAL(wp) :: gkdrag    ! Gravity wave drag coefficient                  (G  in (3), LOTT 1999)
    #  REAL(wp) :: gkwake    ! Bluff-body drag coefficient for low level wake (Cd in (2), LOTT 1999)
    #  REAL(wp) :: gklift    ! Mountain Lift coefficient                      (Cl in (4), LOTT 1999)
    #  !
    #  ! parameters related to the vertical grid
    #  INTEGER  :: nktopg    ! Security value for blocked flow level
    #  INTEGER  :: ntop      ! An estimate to qualify the upper levels of the
    #  !                       model where one wants to impose stress profiles
    #  !
    #  ! scaling with sftlf, the cell area fraction of land incl. lakes
    #  LOGICAL  :: lsftlf    ! true: *lsftlf, false: *1


    # ! "tunable parameters" of the various SSO schemes, same for all domains
    # !
    # REAL(wp), PARAMETER :: gfrcrit = 0.50_wp ! Critical Non-dimensional mountain Height (HNC in (1), LOTT 1999)
    # REAL(wp), PARAMETER :: grcrit  = 0.25_wp ! Critical Richardson Number (Ric, end of first column p791, LOTT 1999)
    # REAL(wp), PARAMETER :: grahilo = 1.00_wp ! Set-up the trapped waves fraction (Beta , end of first column, LOTT 1999)

    # ! numerical security parameters, same for all domains
    # !
    # REAL(wp), PARAMETER :: gsigcr = 0.80_wp      ! Security value for blocked flow depth
    # REAL(wp), PARAMETER :: gssec  = 0.0001_wp    ! Security min value for low-level B-V frequency
    # REAL(wp), PARAMETER :: gtsec  = 0.00001_wp   ! Security min value for anisotropy and GW stress.
    # REAL(wp), PARAMETER :: gvsec  = 0.10_wp      ! Security min value for ulow

    #     ! ECHAM subgrid scale orographic drag configuration
    # ! -------------------------------------------------
    # !
    # ! Define the mask for the SSO parameterization:
    # echam_sso_config(:)% gpicmea = 40.0_wp ! only where  (peak - mean height) is typically > 1st layer depth
    # echam_sso_config(:)% gstd    = 10.0_wp ! only where SSO slope, asymmetry and orientation are defined by EXTPAR
    # !
    # ! Define the tuning parameters for SSO drag. These values depend on:
    # ! (1) the resolution of the topography data used to compute the SSO parameters, and
    # ! (2) the model resolution.
    # ! A 0-value switches the relevant effect off.
    # echam_sso_config(:)% gkdrag  = 0.05_wp
    # echam_sso_config(:)% gkwake  = 0.05_wp
    # echam_sso_config(:)% gklift  = 0.00_wp
    # !
    # ! parameters related to the vertical grid
    # echam_sso_config(:)% ntop    = 1
    # echam_sso_config(:)% nktopg  = 0 ! needs to be derived
    # !
    # ! cell area fraction scaling for SSO effects, .FALSE.: *1, .TRUE.: *sftlf
    # echam_sso_config(:)% lsftlf  = .TRUE.



# Constants:

grav    = 9.80665
rd      = 287.04
cpd     = 1004.64

klev    = 37

# Mask of the SOO parameterization
gpicmea = 0.0       #40.0
gstd    = 0.0       #1.0

ntop    = 0
nktopg  = 36

gfrcrit     = 0.5
grcrit      = 0.25
grahilo     = 1.0
gsigcr      = 0.8
gssec       = 0.0001
gtsec       = 0.00001
gvsec       = 0.1

# Tuning parameters / Switches (if = 0):

gkdrag  = 0.00001#    #1  # 0.05
#gkdrag  = 0.0001    #1  # 0.05
gkwake  = 0         #1 # 0.05
gklift  = 0         #0.002



############################################
# INPUT VARIABLES                          #

# pdtime    length of time step (s)

# paphm1    p at half levels
# papm1     p at full levels
# pmair     air mass (kg/m3)
# ptm1      T
# pum1      u
# pvm1      v

# pmea      Mean orography (m)
# pstd      SSO standard deviation (m)
# psig      SSO slope
# pgam      SSO anisotropy
# pthe      SSO angle
# ppic      SSO peaks elevation (m)
# pcal      SSO valleys elevation (m)
# pftlft    are fraction of land incl. lakes

# pcoriol   Coriolis parameter (1/s), needed for mountain lift
############################################


def calc_SSO(pdtime, zhgeo, paphm1, papm1, pmair, ptm1, pum1, pvm1, pmea, pstd, psig, pgam, pthe, ppic, pval, psftlf, pcoriolis):

    pustrgw, pvstrgw, pvdisgw, pdu_sso, pdv_sso, pdis_sso = sso_drag(pdtime,
                zhgeo, paphm1, papm1, pmair, ptm1, pum1, pvm1,
                pmea, pstd, psig, pgam, pthe, ppic, pval,
                psftlf, pcoriolis)

    return pdu_sso, pdv_sso


####################################################################################################
def sso_drag(pdtime, zhgeo,
                paphm1, papm1, pmair, ptm1, pum1, pvm1,
                pmea, pstd, psig, pgam, pthe, ppic, pval,
                psftlf, pcoriolis): # (29) - Does the SSO drag following Lott & Miller 1997

    pustrgw = 0.0
    pvstrgw = 0.0
    pvdisgw = 0.0

    pdu_sso  = np.zeros(klev)
    pdv_sso  = np.zeros(klev)
    pdis_sso = np.zeros(klev)

    # Select points where scheme is active (187--18)

    if ((ppic - pmea) > gpicmea) and (pstd > gstd):
        # Orographic GWD (208)
        if not (gkwake == 0 and gkdrag == 0):
            zdu_oro, zdv_oro, zdis_oro = orodrag(pdtime,
                                                    zhgeo, paphm1, papm1,
                                                    pmair,
                                                    ptm1, pum1, pvm1,
                                                    pmea, pstd, psig, pgam, pthe, ppic, pval)
            #print('SSO_DRAG // zdu_oro = ' + str(zdu_oro))
            #print('SSO_DRAG // zdv_oro = ' + str(zdv_oro))

        else:
            zdu_oro  = np.zeros(klev)
            zdv_oro  = np.zeros(klev)
            zdis_oro = np.zeros(klev)

        # Mountain lift (225) #TMBNN

        if not gklift == 0:
            zdu_lif, zdv_lif, zdis_lif = orolift(klev, pcoriolis, pdtime, zhgeo,
                                                 paphm1, pmair, ptm1, pum1, pvm1, pmea, pstd, ppic)
        else:
            zdu_lif  = np.zeros(klev)
            zdv_lif  = np.zeros(klev)
            zdis_lif = np.zeros(klev)

    else:
        zdu_oro  = np.zeros(klev)
        zdv_oro  = np.zeros(klev)
        zdis_oro = np.zeros(klev)
        zdu_lif  = np.zeros(klev)
        zdv_lif  = np.zeros(klev)
        zdis_lif = np.zeros(klev)


    # Stress from tendencies [u GW stress, v GW stress, dissipation by GWD] (241)  ##!WWN

    for jk in range(klev):                  # (250)
        pustrgw = pustrgw + ( zdu_oro[jk] + zdu_lif[jk]) * pmair[jk] * psftlf
        pvstrgw = pvstrgw + ( zdv_oro[jk] + zdv_lif[jk]) * pmair[jk] * psftlf
        pvdisgw = pvdisgw + (zdis_oro[jk] +zdis_lif[jk]) * pmair[jk] * psftlf

    # Total quantities [u SSO tendency, v SSO tendency, energy dissipation SSO] (262)

    for jk in range(klev):                  # (267)
        pdu_sso[jk]  = ( zdu_oro[jk] + zdu_lif[jk]) * psftlf
        pdv_sso[jk]  = ( zdv_oro[jk] + zdv_lif[jk]) * psftlf
        pdis_sso[jk] = (zdis_oro[jk] +zdis_lif[jk]) * psftlf

    return pustrgw, pvstrgw, pvdisgw, pdu_sso, pdv_sso, pdis_sso


####################################################################################################
def orodrag(pdtime, phgeo, paphm1, papm1, pmair, ptm1, pum1, pvm1,
                pmea, pstd, psig, pgam, pthe, ppic, pval): # (283) - Does the SSO drag parameterization 
               # (SSO orographically exited waves & low level blcoked flow drag)

    (ikcrit, ikcrith, icrit, ikenvh, iknu, iknu2,
        zrho, zri, zstab, zvph, zpsi, zzdep,
        pulow, pvlow,
        znu, zd1, zd2, zdmod) = orosetup(paphm1, papm1, pmair, pum1, pvm1, ptm1, phgeo,
                                            pthe, pgam, pmea, ppic, pval) # Call setup function (390)

    if not gkdrag == 0:                     # (401)
        # Low level stresses using subcritical and supercritical forms
        # Computes anisotropy coefficient as measure of orographic twodimensionality
        ztau0 = gwstress(ikenvh,
                            pstd, psig, ppic, pval,
                            zrho, zstab, zvph, zdmod)

        # Comupte stress profile including trapped waves, wave breaking, linear decay in stratosphere
        ztau = gwprofil(ikcrith, icrit,                                         
                    pstd,   psig,                                          
                    paphm1, zrho, zri, zstab, zvph, zdmod,
                    ztau0)
        # Inside gwprofil, ptau is pointing to ztau, and this has INTENT(out);
        # additionally, new ztau in gwprofil
    
    # Tendencies from waves stress profile (428)
    # Low level blocked flow drag #TMBNN
    zdudt = 0.0
    zdvdt = 0.0

    pvom = np.zeros(klev)
    pvol = np.zeros(klev)
    pdis = np.zeros(klev)

    #print('ORODRAG // ztau = ' + str(ztau))
    #print('ORODRAG // zvph = ' + str(zvph))    
    #print('ORODRAG // pmair = ' + str(pmair))




    # Wave stress (470)

    # ztau is the stress profile from gwprofil, zvph is wind in plan of GW stress, half levels
    # zdmod is norm of zd1, zd2 (define plane of low level stress compared to low level wind)
    

    for jk in range(klev):                  # (457)

        if not gkdrag == 0:
            ztemp = - (ztau[jk+1] - ztau[jk]) / (zvph[klev] * pmair[jk])
            zdudt = (pulow * zd1 - pvlow * zd2) * ztemp / zdmod
            zdvdt = (pvlow * zd1 - pulow * zd2) * ztemp / zdmod

            # Control overshoots (479)

            zforc = np.sqrt(zdudt**2 + zdvdt**2)
            ztend = np.sqrt(pum1[jk]**2 + pvm1[jk]**2) / pdtime
            rover = 0.25

            if zforc >= rover*ztend:
                zdudt = rover * ztend / zforc * zdudt
                zdvdt = rover * ztend / zforc * zdvdt
        
        else:
            zdudt = 0.0
            zdvdt = 0.0

            
#### THIS WAS COMMENTED IN V1
            # Blocked flow drag (497) #TMBNN
        if jk > ikenvh:
            if not gkwake == 0.0:

                zb      = 1.0 - 0.18 * pgam - 0.04 * pgam**2
                zc      = 0.48 * pgam + 0.30 * pgam**2
                zconb   = 2.0 * pdtime * gkwake * psig / (4.0 * pstd)
                zabsv   = np.sqrt(pum1[jk]**2 + pvm1[jk]**2) / 2.0
                zzd1    = zb * np.cos(zpsi[jk])**2 + zc * np.sin(zpsi[jk])**2
                ratio   = (np.cos(zpsi[jk])**2 + pgam * np.sin(zpsi[jk])**2) / (pgam * np.cos(zpsi[jk])**2 +np.sin(zpsi[jk])**2)
                zbet    = max(0.0, 2.0 - 1.0 / ratio) * zconb *zzdep[jk] * zzd1 * zabsv

                    # Opposed to the wind

                zdudt = - pum1[jk] / pdtime
                zdvdt = - pvm1[jk] / pdtime

                zdudt = zdudt*(zbet/(1.0+zbet))
                zdvdt = zdvdt*(zbet/(1.0+zbet))

            else:
                zdudt = 0.0
                zdvdt = 0.0
#### END OF COMMENT V1


        # Tendencies:
        pvom[jk]     = zdudt
        pvol[jk]     = zdvdt

        zust            = pum1[jk] + pdtime * zdudt
        zvst            = pvm1[jk] + pdtime * zdvdt
        zdis            = 0.5 * (pum1[jk]**2 + pvm1[jk]**2 - zust**2 - zvst**2)

        if zdis < 0.0:
            zred        = np.sqrt((pum1[jk]**2 + pvm1[jk]**2) / (zust**2 + zvst**2))
            zust        = zust * zred
            zvst        = zvst * zred
            pvom[jk]    = (zust - pum1[jk]) / pdtime
            pvol[jk]    = (zvst - pvm1[jk]) / pdtime
            zdis        = 0.5 * (pum1[jk]**2 + pvm1[jk]**2 - zust**2 - zvst**2)

        pdis[jk]     = zdis / pdtime

    #print('ORODRAG // pvom = ' + str(pvom))
    #print('ORODRAG // pvol = ' + str(pvol))
    #print('ORODRAG // pdis = ' + str(pdis))


    return pvom, pvol, pdis


####################################################################################################
def orosetup(paphm1, papm1, pmair, pum1, pvm1, ptm1, phgeo,
                ptheta, pgam, pmea, ppic, pval):
                # (560) - Set-up the essential parameters of the SSO drag scheme:
                #         Depth of low blocked layer, low-level flow, background stratification

    # Out:

    prho    = np.zeros(klev+1)
    pri     = np.zeros(klev+1)
    pstab   = np.zeros(klev+1)
    #ptau    = np.zeros(klev+1)
    pvph    = np.zeros(klev+1)
    ppsi    = np.zeros(klev+1)
    pzdep   = np.zeros(klev)


    # Local:

    zhcrit  = np.zeros(klev)
    zdp     = np.zeros(klev)
    zmair   = 0.0


    # Computational constants: (695)
    ilevh  = math.floor(klev/3)
    zcons1 = 1/rd
    zcons2 = grav**2/cpd

    kknu    = klev
    kknu2   = klev
    kknub   = klev
    kknul   = klev
    pgam    = max(pgam, gtsec)

    ll1     = np.full(klev+1, False, dtype=bool)
    for jk in range(klev, ilevh, -1):
        ll1[jk] = True


    # Define low-level wind, projet winds in plane of low-level wind, determine sector in which
    # to take variance and set indicator for critical levels

    for jk in range(klev-1, ilevh-1, -1):       # (736)
        lo = paphm1[jk]/paphm1[klev] >= gsigcr

        if lo:
            kkcrit = jk

        zhcrit[jk] = ppic - pval
        ll1[jk]    = (phgeo[jk] > zhcrit[jk])

        if ll1[jk] is not ll1[jk+1]:
            kknu = jk
        if not ll1[ilevh]:
            kknu = ilevh


    for jk in range(klev-1, ilevh-1, -1):       # (762)
        zhcrit[jk] = ppic - pmea
        ll1[jk]    = (phgeo[jk] > zhcrit[jk])

        if ll1[jk] is not ll1[jk+1]:
            kknu2 = jk
        if not ll1[ilevh]:
            kknu2 = ilevh


    for jk in range(klev-1, ilevh-1, -1):       # (784)
        zhcrit[jk]  = min(ppic - pmea, pmea - pval)
        ll1[jk]     = (phgeo[jk] > zhcrit[jk])

        if ll1[jk] is not ll1[jk+1]:
            kknub = jk
        if not ll1[ilevh]:
            kknub = ilevh


    kknu  = min(kknu, nktopg)
    kknu2 = min(kknu2, nktopg)
    kknub = min(kknub, nktopg)
    kknul = klev


    # ... initialize various arrays ...

    kkenvh      = klev
    kkcrith     = klev
    kcrit       = 0 # (this was 1 in FORTRAN -> 0)

    # Define flow density and stratification (rho and N2) at semi layers (834)

    for jk in range(klev-1,0,-1):                # (840, here index until 2 -> 1)
        zdp[jk]   = papm1[jk]- papm1[jk-1]
        prho[jk]  = 2.0 * paphm1[jk] * zcons1 / (ptm1[jk] + ptm1[jk-1])
        pstab[jk] = (2.0 * zcons2 / (ptm1[jk] + ptm1[jk-1])
                     * (1.0 - cpd * prho[jk] *(ptm1[jk] - ptm1[jk-1]) / zdp[jk]))
        pstab[jk] = max(pstab[jk], gssec)


    # Define low-level flow (between ground and peaks-valleys) (862)

    pulow = 0.0
    pvlow = 0.0

    for jk in range(klev-1, ilevh-1, -1):           # (866)
        if jk >= kknu2 and jk <= kknul:
            pulow           = pulow         + pum1[jk]   * pmair[jk]
            pvlow           = pvlow         + pvm1[jk]   * pmair[jk]
            pstab[klev]   = pstab[klev] + pstab[jk]  * pmair[jk]
            prho[klev]    = prho[klev]  + prho[jk]   * pmair[jk]
            zmair           = zmair         +              pmair[jk]

 
    zmair_inv        = 1.0 / zmair
    pulow           = pulow         * zmair_inv
    pvlow           = pvlow         * zmair_inv
    pstab[klev]   = pstab[klev] * zmair_inv
    prho[klev]    = prho[klev]  * zmair_inv

    znorm        = max(np.sqrt(pulow**2 + pvlow**2), gvsec)
    pvph[klev] = znorm


    # Setup orography relative to the low-level wind and define parameters
    # for the anisotropic wave stress (903)
  
    lo = (pulow < gvsec) and (pulow >= -gvsec)
    
    if lo:
        zu = pulow + 2.0 * gvsec
    else:
        zu = pulow

    zphi            = np.arctan(pvlow / zu)
    ppsi[klev]    = ptheta * math.pi / 180.0 - zphi
    zb              = 1.0 - 0.18 * pgam - 0.04 * pgam**2
    zc              = 0.48 * pgam + 0.3 * pgam**2
    pd1             = zb - (zb - zc) * (np.sin(ppsi[klev])**2)
    pd2             = (zb - zc) * np.sin(ppsi[klev]) * np.cos(ppsi[klev])
    pdmod           = np.sqrt(pd1**2 + pd2**2)


    # Project flow in plane lof low-level stress
    # Find critical levels (928)

    zvpf = np.zeros(klev)

    for jk in range(klev):                  # (933)
        zvt1        = pulow * pum1[jk] + pvlow * pvm1[jk]
        zvt2        = - pvlow * pum1[jk] + pulow *pvm1[jk]
        zvpf[jk] = (zvt1 * pd1 + zvt2 * pd2) / (znorm * pdmod)


    for jk in range(klev):                  # (951)
        pzdep[jk]    = 0.0
        ppsi[jk]     = 0.0
        ll1[jk]      = 0.0

    ppsi[klev] = 0.0
    ll1[klev]  = 0.0

    for jk in range(0,klev):                # (971, here index from 2 -> 1)
        zdp[jk]  = papm1[jk] - papm1[jk-1]
        pvph[jk] = (((paphm1[jk] - papm1[jk-1]) * zvpf[jk] +
                        (papm1[jk] - paphm1[jk] ) * zvpf[jk-1])
                        / zdp[jk])

        if pvph[jk] < gvsec:
            pvph[jk] = gvsec

            if jk < klev:
                kcrit = jk


    # Mean flow Richardson number (993)

    for jk in range(0,klev):                # (997, index from 2 -> 1)
        zdwind      = max(abs(zvpf[jk] - zvpf[jk-1]), gvsec)
        pri[jk]  = pstab[jk] * (zdp[jk] / (grav * prho[jk] * zdwind))**2
        pri[jk]  = max(pri[jk], grcrit)


    # Define top of envelope layer (1014)

    pnu     = 0.0
    znum    = 0.0

    for jk in range(0, klev-2):             # (1026, index from 2 -> 1)
        if jk >= kknu:
            znum        = pnu
            zwind       = ((pulow * pum1[jk] + pvlow * pvm1[jk]) /
                            max(np.sqrt(pulow**2 + pvlow**2), gvsec))
            zwind       = max(np.sqrt(zwind**2), gvsec)
            zstabm      = np.sqrt(max(pstab[jk], gssec))
            zstabp      = np.sqrt(max(pstab[jk+1], gssec))
            zrhom       = prho[jk]
            zrhop       = prho[jk+1]
            pnu         = pnu + pmair[jk] * ((zstabp / zrhop + zstabm / zrhom) / 2.0) / zwind

            if znum <= gfrcrit and pnu > gfrcrit and kkenvh == klev:
                kkenvh = jk



    # Calculation of a dynamical mixing height for when the waves break at low level:
    # The drag will be repartited over a depth that depends on waves vertical wavelength, not just
    # between two adjacent model layers (1054)

    znup    = 0.0                       
    znum    = 0.0

    for jk in range(klev-2,0,-1):               # (1070, index to 2 -> 1)
        znum        = znup
        zwind       = ((pulow * pum1[jk] + pvlow * pvm1[jk]) /
                        max(np.sqrt(pulow**2 + pvlow**2), gvsec))
        zwind       = max(np.sqrt(zwind**2), gvsec)
        zstabm      = np.sqrt(max(pstab[jk], gssec))
        zstabp      = np.sqrt(max(pstab[jk+1], gssec))
        zrhom       = prho[jk]
        zrhop       = prho[jk+1]
        znum        = znup + pmair[jk] * ((zstabp / zrhop + zstabm / zrhom) / 2.0) / zwind

        if znum <= math.pi / 2.0 and znup > math.pi / 2.0 and kkcrith == 0:
            kkcrith = jk

 
    kkcrith     = min(kkcrith, kknu)
    kkcrith     = max(kkcrith, ilevh*2)
    if kcrit >= kkcrith:
        kcrit = 1


    # Directional info for flow blocking (1105)    

    for jk in range(klev):                      # (1109)
        lo = (pum1[jk] < gvsec) and (pum1[jk] >= -gvsec)
        if lo:
            zu = pum1[jk] + 2.0 * gvsec
        else:
            zu = pum1[jk]
        
        zphi        = np.arctan(pvm1[jk] / zu)
        ppsi[jk]    = ptheta * math.pi / 180.0 - zphi


    # Forms the vertical 'leakiness' (1131)

        for jk in range(ilevh-1, klev-1):           # (1135)
            pzdep[jk] = 0.0

            if jk >= kkenvh and kkenvh != klev:
                pzdep[jk] = (((phgeo[kkenvh]-1) - phgeo[jk]) /
                                (phgeo[kkenvh] - phgeo[klev-1]))


    return (kkcrit, kkcrith, kcrit, kkenvh, kknu, kknu2,
            prho, pri, pstab, pvph, ppsi, pzdep,
            pulow, pvlow,
            pnu, pd1, pd2, pdmod)


####################################################################################################
def gwstress(kkenvh, pstd, psig, ppic, pval, prho, pstab, pvph, pdmod):     # Compute the surface stress due to gravity waves, according to the Phillips
                    # 1979 theory of 3D flow above anisotropic elliptic ridges
                    # The stress is reduced to account for cut-off flow over hill. The flow only
                    # sees that part of the ridge located above the blocked layer. (1153)


    # Gravity wave stress (1207)

    ptau0 = 0.0

    # Effective mountain height above the blocked flow (1223)

    zeff = ppic - pval

    if kkenvh < klev:
        zeff = min(gfrcrit * pvph[klev] / np.sqrt(pstab[klev]), zeff)

    ptau0 = (gkdrag  * prho[klev]
                    * psig * pdmod / 4.0 / pstd
                    * pvph[klev] * np.sqrt(pstab[klev])
                    * zeff**2)

    return ptau0


####################################################################################################
def gwprofil(kkcrith, kcrit,
                pstd,   psig,
                paphm1, prho, pri, pstab,  pvph, pdmod,
                ptau0):     # The stress profile for gravity waves is computed as follows:
                    # It decreases linearly with heights from the ground to the low-level indicated
                    # by kkcrith, to simulate lee waves of low-level gravity wave breaking. Above it
                    # is constant, except when the waves encounter a critical level kcrit or when
                    # they break. The stress is also uniformly distributed above the level ntop.
                    # (1152)

    ztau = np.zeros(klev+1) # Local!!
    ptau = np.zeros(klev+1) # This is the global ztau and will be returned to the main program


    zoro            = psig * pdmod / 4.0 / pstd
    ztau[klev]    = ptau0
    ztau[kkcrith]   = grahilo * ptau0

    
    # Constant shear stress until top of the low-level breaking / trapped layer (1342)

    for jk in range(klev,0,-1):               # (1339, index until 2 -> 1)
        if jk > kkcrith:
            zdelp       = paphm1[jk] - paphm1[klev]
            zdelpt      = paphm1[kkcrith] - paphm1[klev]
            ptau[jk]    = (ztau[klev] + zdelp / zdelpt *
                            (ztau[kkcrith] - ztau[klev]))
        else:
            ptau[jk] = ztau[kkcrith]

    
    # Constant shear stress until the top of the low level flow layer (1370)
    # Wave displacement at next level (1375)
    # Wave Richardson number, new wave displacement and stress: Breaking evaluation
    #       critical level (1380)

    zdz2 = np.zeros(klev)

    for jk in range(klev-1,0,-1):                 # (1386, index until 2 -> 1)
        znorm       = prho[jk] * np.sqrt(pstab[jk]) * pvph[jk]
        zdz2[jk]    = ptau[jk] / max(znorm, gssec) / zoro
  
        if jk < kkcrith:
            if ptau[jk+1] < gtsec or jk <= kcrit:
                ptau[jk] = 0.0
            else:
                zsqr        = np.sqrt(pri[jk])
                zalfa       = np.sqrt(pstab[jk] * zdz2[jk]) / pvph[jk]
                zriw        = pri[jk] * (1.0 - zalfa) / (1 + zalfa * zsqr)**2
                if zriw < grcrit:
                    zdel        = 4.0 / zsqr / grcrit + 1 / grcrit**2 + 4 / grcrit
                    zb          = 1.0 / grcrit + 2.0 / zsqr
                    zalpha      = 0.5 * ( -zb + np.sqrt(zdel))
                    zdz2n       = (pvph[jk] * zalpha)**2 / pstab[jk]
                    ptau[jk] = znorm * zdz2n * zoro
                
                ptau[jk] = min(ptau[jk], ptau[jk+1])


        # Reorganisation of the stress profile at low level (1433)

        for jk in range(klev):                  # (1451)
            if jk > kkcrith:
                zdelp       = paphm1[jk] - paphm1[klev]
                zdelpt      = paphm1[kkcrith] - paphm1[klev]
                ptau[jk]    = ztau[klev] + (ztau[kkcrith] - ztau[klev]) * zdelp / zdelpt

        
        # Reorganisation at the model top (1473)

        if jk < ntop:
            ptau[jk]     = ztau[ntop]

    return ptau



####################################################################################################
def orolift(klev, pcoriol, pdtime, zhgeo, paphm1, pmair, ptm1, pum1, pvm1, pmea, pstd, ppic):          # Simulate the geostrophic lift.
                        # Computes the physical tendencies of the prognostic variables u, v
                        # when enhanced vortex stretching is needed. (1503)

    # Initializations (1589)

    lifthigh = False

    zcons1      = 1.0 / rd

    iknub           = klev
    iknul           = klev
    
    zrho    = np.zeros(klev+1)
    ztau    = np.zeros(klev+1)
    ztav    = np.zeros(klev+1)

    ll1     = np.full(klev+1, False, dtype=bool)

    zmair           = 0.0
    pulow           = 0.0
    pvlow           = 0.0

    pvom = np.zeros(klev)
    pvol = np.zeros(klev)
    pdis = np.zeros(klev)

    zhcrit = np.zeros(klev)


    
    # Define low level wind, project winds in plane of low level wind, determine sector in which
    # to take the vairance and set indicator for critical levels. (1623)


    for jk in range(klev-1,0,-1):                 # (1630)
        zhcrit[jk]   = max(ppic - pmea, 100.0)
        ll1[jk]      = zhgeo[jk] > zhcrit[jk]

        if ll1[jk] is not ll1[jk+1]:
            iknub   = jk

    iknub   = max(iknub, klev/2)
    iknul   = max(iknul, 2*klev/3)

    if iknub > nktopg:
        iknub   = nktopg
    if iknub == nktopg:
        iknul   = klev
    if iknub == iknul:
        iknub   = iknul - 1

    for jk in range(klev-1,1,-1):                 # (1659)
        zrho[jk] = 2.0 * paphm1[jk] * zcons1 / (ptm1[jk] + ptm1[jk-1])

    
    # Define low level flow (1670)

    for jk in range(klev-1,0,-1):                 # (1674)
        if jk >= iknub and jk <= iknul:
            pulow           = pulow         + pum1[jk] * pmair[jk]
            pvlow           = pvlow         + pvm1[jk] * pmair[jk]
            zrho[klev]      = zrho[klev]    + zrho[jk] * pmair[jk]
            zmair           = zmair         +            pmair[jk]
                
    zmair_inv       = 1.0 / zmair
    pulow           = pulow      * zmair_inv
    pvlow           = pvlow      * zmair_inv
    zrho[klev]      = zrho[klev] * zmair_inv


    # Compute mountain lift (1705)

    ztau[klev]    = (-gklift * zrho[klev] * pcoriol *
                        2.0 * pstd * pvlow)
    ztav[klev]    =  (gklift * zrho[klev] * pcoriol *
                            2.0 * pstd * pulow)
  
    
    # Compute lift profile (1728)

    for jk in range(klev):                      # (1734)
        ztau[jk] = ztau[klev] * paphm1[jk] / paphm1[klev]
        ztav[jk] = ztav[klev] * paphm1[jk] / paphm1[klev]
     
    
    # Compute tendencies (1749)

    if lifthigh:

        # Explicit solutions at all levels
        for jk in range(klev):                  # (1759)
            zmair_inv = 1.0 / pmair[jk]
            zdudt = -(ztau[jk+1] - ztau[jk]) * zmair_inv
            zdvdt = -(ztav[jk+1] - ztav[jk]) * zmair_inv

        # Project perpendicularly to u not to destroy energy
        for jk in range(klev):                  # (1775)
            zslow   = np.sqrt(pulow**2 + pvlow**2)
            zsqua   = max(np.sqrt(pum1[jk]**2 + pvm1[jk]**2), gvsec)
            zscav   = - zdudt * pvm1[jk] + zdvdt * pum1[jk]

            if zsqua > gvsec:
                pvom[jk] = - zscav * pvm1[jk] / zsqua**2
                pvol[jk] =   zscav * pum1[jk] / zsqua**2
            else:
                pvom[jk] = 0.0
                pvol[jk] = 0.0
            
            zsqua   = np.sqrt(pum1[jk]**2 + pum1[jk]**2)

            if zsqua < zslow:
                pvom[jk] = zsqua / zslow * pvom[jk]
                pvol[jk] = zsqua / zslow * pvol[jk]
        

    # Low level lift, semi implicit (1799)

    for jk in range(klev-1,iknub,-1): # (1809)
        zbet        = (gklift * pcoriol * pdtime *
                        (zhgeo[iknub-1] - zhgeo[jk]) /
                        (zhgeo[iknub-1] - zhgeo[klev-1]))
        zdudt   = - pum1[jk] / pdtime / (1+zbet**2)
        zdvdt   = - pvm1[jk] / pdtime / (1+zbet**2)
        pvom[jk] = zbet**2 * zdudt - zbet * zdvdt
        pvol[jk] = zbet * zdudt + zbet**2 * zdvdt

    return pvom, pvol, pdis
# EOF