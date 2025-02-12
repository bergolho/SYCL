#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
    const double natp = 0.24; // K dependence of ATP-sensitive K current
    const double nicholsarea = 0.00005; // Nichol's areas (cm^2)
    const double hatp = 2;             // Hill coefficient

    //Linear changing of atpi depending on the fibrosis and distance from the center of the scar (only for border zone cells)
    double atpi = extra_parameters[0];
    double atpi_change = 6.8f - atpi;
    atpi = atpi + atpi_change*fibrosis;

    //Extracellular potassium concentration was elevated
    //from its default value of 5.4 mM to values between 6.0 and 8.0 mM
    //Ref: A Comparison of Two Models of Human Ventricular Tissue: Simulated Ischemia and Re-entry
    double Ko = extra_parameters[1];
    double Ko_change  = 5.4f - Ko;
    Ko = Ko + Ko_change*fibrosis;

    double Ki = extra_parameters[2];
    double Ki_change  = 138.3 - Ki;
    Ki = Ki + Ki_change*fibrosis;  

    double Vm_modifier = extra_parameters[3];
    Vm_modifier = Vm_modifier - Vm_modifier*fibrosis;

    double GNa_multplicator = extra_parameters[4];
    double GNa_multplicator_change  = 1.0f - GNa_multplicator;
    GNa_multplicator = GNa_multplicator + GNa_multplicator_change*fibrosis;

    double GCaL_multplicator = extra_parameters[5];
    double GCaL_multplicator_change  = 1.0f - GCaL_multplicator;
    GCaL_multplicator = GCaL_multplicator + GCaL_multplicator_change*fibrosis;

    double INaCa_multplicator = extra_parameters[6];
    double INaCa_multplicator_change  = 1.0f - INaCa_multplicator;
    INaCa_multplicator = INaCa_multplicator + INaCa_multplicator_change*fibrosis;

    double Ikatp_multiplicator = extra_parameters[7];
    double Ikatp_multiplicator_change  = 1.0f - Ikatp_multiplicator;
    Ikatp_multiplicator = Ikatp_multiplicator + Ikatp_multiplicator_change*fibrosis;
     

    //double katp = 0.306;
    //Ref: A Comparison of Two Models of Human Ventricular Tissue: Simulated Ischaemia and Re-entry
    //double katp = 0.306;
    const double katp = -0.0942857142857*atpi + 0.683142857143; //Ref: A Comparison of Two Models of Human Ventricular Tissue: Simulated Ischaemia and Re-entry

    const double patp = 1 / (1 + dpct::pow((atpi / katp), hatp));
    const double gkatp    =  0.000195/nicholsarea;
    const double gkbaratp = gkatp * patp * dpct::pow((Ko / 5.4), natp);

    const double katp2= 1.4;
    const double hatp2 = 2.6;
    const double pcal = 1.0 / (1.0 + dpct::pow((katp2 / atpi), hatp2));

    const double Cao=2.0;
    const double Nao=140.0;
    const double Cai=0.00007;
    const double Nai=7.67;

//Constants
    const double R=8314.472;
    const double F=96485.3415;
    const double T=310.0;
    const double RTONF=(R*T)/F;

//Parameters for currents
//Parameters for IKr
    const double Gkr=0.101;
//Parameters for Iks
    const double pKNa=0.03;
#ifdef EPI
    const double Gks=0.257;
#endif
#ifdef ENDO
    const double Gks=0.392;
#endif
#ifdef MCELL
    const double Gks=0.098;
#endif
//Parameters for Ik1
    const double GK1=5.405;
//Parameters for Ito
#ifdef EPI
    const double Gto=0.294;
#endif
#ifdef ENDO
    const double Gto=0.073;
#endif
#ifdef MCELL
    const double Gto=0.294;
#endif
//Parameters for INa
    const double GNa=14.838*GNa_multplicator; //ACIDOSIS
//Parameters for IbNa
    const double GbNa=0.00029;
//Parameters for INaK
    const double KmK=1.0;
    const double KmNa=40.0;
    const double knak=2.724;
//Parameters for ICaL
    const double GCaL=0.2786*pcal*GCaL_multplicator; //ACIDOSIS
//Parameters for IbCa
    const double GbCa=0.000592;
//Parameters for INaCa
    const double knaca=1000;
    const double KmNai=87.5;
    const double KmCa=1.38;
    const double ksat=0.1;
    const double n=0.35;
//Parameters for IpCa
    const double GpCa=0.1238;
    const double KpCa=0.0005;
//Parameters for IpK;
    const double GpK=0.0293;

    const double Ek = RTONF * (sycl::log((Ko / Ki)));
    const double Ena = RTONF * (sycl::log((Nao / Nai)));
    const double Eks = RTONF * (sycl::log((Ko + pKNa * Nao) / (Ki + pKNa * Nai)));
    const double Eca = 0.5 * RTONF * (sycl::log((Cao / Cai)));
    double IKr;
    double IKs;
    double IK1;
    double Ito;
    double INa;
    double IbNa;
    double ICaL;
    double IbCa;
    double INaCa;
    double IpCa;
    double IpK;
    double INaK;
    double IKatp;

    double Ak1;
    double Bk1;
    double rec_iK1;
    double rec_ipK;
    double rec_iNaK;
    double AM;
    double BM;
    double AH_1;
    double BH_1;
    double AH_2;
    double BH_2;
    double AJ_1;
    double BJ_1;
    double AJ_2;
    double BJ_2;
    double M_INF;
    double H_INF;
    double J_INF;
    double TAU_M;
    double TAU_H;
    double TAU_J;
    double axr1;
    double bxr1;
    double Xr1_INF;
    double Xr2_INF_new;
    double TAU_Xr1;
    double Axs;
    double Bxs;
    double Xs_INF;
    double TAU_Xs;
    double R_INF_new;
    double S_INF;
    double TAU_S;
    double Af;
    double Bf;
    double Cf;
    double Af2;
    double Bf2;
    double Cf2;
    double D_INF_new;
    double TAU_F;
    double F_INF;
    double TAU_F2;
    double F2_INF;
    double sItot;


    //Needed to compute currents
    Ak1 = 0.1 / (1. + sycl::exp(0.06 * (svolt - Ek - 200)));
    Bk1 = (3. * sycl::exp(0.0002 * (svolt - Ek + 100)) +
           sycl::exp(0.1 * (svolt - Ek - 10))) /
          (1. + sycl::exp(-0.5 * (svolt - Ek)));
    rec_iK1=Ak1/(Ak1+Bk1);
    rec_iNaK = (1. / (1. + 0.1245 * sycl::exp(-0.1 * svolt * F / (R * T)) +
                      0.0353 * sycl::exp(-svolt * F / (R * T))));
    rec_ipK = 1. / (1. + sycl::exp((25 - svolt) / 5.98));

    //Compute currents
    INa=GNa*sm*sm*sm*sh*sj*((svolt-Vm_modifier)-Ena); //ACIDOSIS
    ICaL=GCaL*D_INF*sf*sf2*((svolt-Vm_modifier)-60); //ACIDOSIS
    Ito=Gto*R_INF*ss*(svolt-Ek);
    IKr = Gkr * sycl::sqrt(Ko / 5.4) * sxr1 * Xr2_INF * (svolt - Ek);
    IKs=Gks*sxs*sxs*(svolt-Eks);
    IK1=GK1*rec_iK1*(svolt-Ek);
    INaCa = knaca * (1. / (KmNai * KmNai * KmNai + Nao * Nao * Nao)) *
            (1. / (KmCa + Cao)) *
            (1. / (1 + ksat * sycl::exp((n - 1) * svolt * F / (R * T)))) *
            (sycl::exp(n * svolt * F / (R * T)) * Nai * Nai * Nai * Cao -
             sycl::exp((n - 1) * svolt * F / (R * T)) * Nao * Nao * Nao * Cai *
                 2.5);

    INaCa = INaCa*INaCa_multplicator; //ACIDOSIS

    INaK=knak*(Ko/(Ko+KmK))*(Nai/(Nai+KmNa))*rec_iNaK;
    IpCa=GpCa*Cai/(KpCa+Cai);
    IpK=GpK*rec_ipK*(svolt-Ek);
    IbNa=GbNa*(svolt-Ena);
    IbCa=GbCa*(svolt-Eca);

    IKatp = gkbaratp*(svolt-Ek) * Ikatp_multiplicator;


    //Determine total current
    (sItot) = IKr    +
              IKs   +
              IK1   +
              Ito   +
              INa   +
              IbNa  +
              ICaL  +
              IbCa  +
              INaK  +
              INaCa +
              IpCa  +
              IpK   +
              IKatp +
              stim_current;

    //compute steady state values and time constants
    AM = 1. / (1. + sycl::exp((-60. - svolt) / 5.));
    BM = 0.1 / (1. + sycl::exp((svolt + 35.) / 5.)) +
         0.10 / (1. + sycl::exp((svolt - 50.) / 200.));
    TAU_M=AM*BM;
    M_INF = 1. / ((1. + sycl::exp((-56.86 - svolt) / 9.03)) *
                  (1. + sycl::exp((-56.86 - svolt) / 9.03)));
    if (svolt>=-40.)
    {
        AH_1=0.;
        BH_1 = (0.77 / (0.13 * (1. + sycl::exp(-(svolt + 10.66) / 11.1))));
        TAU_H= 1.0/(AH_1+BH_1);
    }
    else
    {
        AH_2 = (0.057 * sycl::exp(-(svolt + 80.) / 6.8));
        BH_2 = (2.7 * sycl::exp(0.079 * svolt) +
                (3.1e5) * sycl::exp(0.3485 * svolt));
        TAU_H=1.0/(AH_2+BH_2);
    }
    H_INF = 1. / ((1. + sycl::exp((svolt + 71.55) / 7.43)) *
                  (1. + sycl::exp((svolt + 71.55) / 7.43)));
    if(svolt>=-40.)
    {
        AJ_1=0.;
        BJ_1 = (0.6 * sycl::exp((0.057) * svolt) /
                (1. + sycl::exp(-0.1 * (svolt + 32.))));
        TAU_J= 1.0/(AJ_1+BJ_1);
    }
    else
    {
        AJ_2 = (((-2.5428e4) * sycl::exp(0.2444 * svolt) -
                 (6.948e-6) * sycl::exp(-0.04391 * svolt)) *
                (svolt + 37.78) / (1. + sycl::exp(0.311 * (svolt + 79.23))));
        BJ_2 = (0.02424 * sycl::exp(-0.01052 * svolt) /
                (1. + sycl::exp(-0.1378 * (svolt + 40.14))));
        TAU_J= 1.0/(AJ_2+BJ_2);
    }
    J_INF=H_INF;

    Xr1_INF = 1. / (1. + sycl::exp((-26. - svolt) / 7.));
    axr1 = 450. / (1. + sycl::exp((-45. - svolt) / 10.));
    bxr1 = 6. / (1. + sycl::exp((svolt - (-30.)) / 11.5));
    TAU_Xr1=axr1*bxr1;
    Xr2_INF_new = 1. / (1. + sycl::exp((svolt - (-88.)) / 24.));

    Xs_INF = 1. / (1. + sycl::exp((-5. - svolt) / 14.));
    Axs = (1400. / (sycl::sqrt(1. + sycl::exp((5. - svolt) / 6))));
    Bxs = (1. / (1. + sycl::exp((svolt - 35.) / 15.)));
    TAU_Xs=Axs*Bxs+80;

#ifdef EPI
    R_INF_new=1./(1.+exp((20-svolt)/6.));
    S_INF=1./(1.+exp((svolt+20)/5.));
    TAU_S=85.*exp(-(svolt+45.)*(svolt+45.)/320.)+5./(1.+exp((svolt-20.)/5.))+3.;
#endif
#ifdef ENDO
    R_INF_new = 1. / (1. + sycl::exp((20 - svolt) / 6.));
    S_INF = 1. / (1. + sycl::exp((svolt + 28) / 5.));
    TAU_S = 1000. * sycl::exp(-(svolt + 67) * (svolt + 67) / 1000.) + 8.;
#endif
#ifdef MCELL
    R_INF_new=1./(1.+exp((20-svolt)/6.));
    S_INF=1./(1.+exp((svolt+20)/5.));
    TAU_S=85.*exp(-(svolt+45.)*(svolt+45.)/320.)+5./(1.+exp((svolt-20.)/5.))+3.;
#endif

    D_INF_new = 1. / (1. + sycl::exp((-8 - svolt) / 7.5));
    F_INF = 1. / (1. + sycl::exp((svolt + 20) / 7));
    Af = 1102.5 * sycl::exp(-(svolt + 27) * (svolt + 27) / 225);
    Bf = 200. / (1 + sycl::exp((13 - svolt) / 10.));
    Cf = (180. / (1 + sycl::exp((svolt + 30) / 10))) + 20;
    TAU_F=Af+Bf+Cf;
    F2_INF = 0.67 / (1. + sycl::exp((svolt + 35) / 7)) + 0.33;
    Af2 = 600 * sycl::exp(-(svolt + 27) * (svolt + 27) / 170);
    Bf2 = 7.75 / (1. + sycl::exp((25 - svolt) / 10));
    Cf2 = 16 / (1. + sycl::exp((svolt + 30) / 10));
    TAU_F2=Af2+Bf2+Cf2;

    //update voltage
    rDY_[0] = -sItot;

    //Update gates
    rDY_[1] = M_INF - (M_INF - sm) * sycl::exp(-dt / TAU_M);
    rDY_[2] = H_INF - (H_INF - sh) * sycl::exp(-dt / TAU_H);
    rDY_[3] = J_INF - (J_INF - sj) * sycl::exp(-dt / TAU_J);
    rDY_[4] = Xr1_INF - (Xr1_INF - sxr1) * sycl::exp(-dt / TAU_Xr1);
    rDY_[5] = Xs_INF - (Xs_INF - sxs) * sycl::exp(-dt / TAU_Xs);
    rDY_[6] = S_INF - (S_INF - ss) * sycl::exp(-dt / TAU_S);
    rDY_[7] = F_INF - (F_INF - sf) * sycl::exp(-dt / TAU_F);
    rDY_[8] = F2_INF - (F2_INF - sf2) * sycl::exp(-dt / TAU_F2);

    rDY_[9] = D_INF_new;
    rDY_[10] = R_INF_new;
    rDY_[11] = Xr2_INF_new;