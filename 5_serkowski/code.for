       PROGRAM SERK_OBS_2
      ! PROGRAM SERK_OBS_2 (NON-Linear Approx)
      !
      ! Serkowski law fitting of a data table {lambda(i), P(i), sigma_P(i)}
      ! by the Levenberg-Marquardt method of nonlinear approximation. 
      ! with  K = a * lambda_max + b from Wilking+ 1982; Whittet+ 1992.
      !
      ! Uses the Numerical Recipes subroutines:
      !    MRQMIN (basic), MRQCOF, COVSRT, GAUSSJ. 
      ! How they work is explained in the Num.Rec. Sect.15.
      !
      ! Important: Single precision code!   вв
      !
      ! INPUT:
      !   Line 1: 
      !      the object name STAR (1x, A8) and  
      !      the number nLam0 (I4) of lambda(i) points (nLam0 < NPT < 101)
      !
      !   Lines 2-...:
      !      {Lam(i),P(i)} for i = 1, nLam0 
      !          are taken from the file 'inp.txt' 
      !          by using the format (1x, 3F6.3)
      !   
      !   One should set some GUESS values for P_max & lambda_max (gPm & gLm).
      !      It is made in the code by taking max{P(i)} & its lambda from input.
      !   Note that if these values are relatively far from the true ones,
      !        iterations in the MRGMIN will not converge!
      !
      !   The approximation function is defined by the external 
      !     subroutine FUNCS (X,A,Y,DYDA,NA), where: 
      !       X = 1/lambda,
      !       A(NA) are the parameters of Serkowski curve: 
      !         A(1) = P_max,  A(2) = lambda_max,
      !       Y is the value of the approximation function at X,
      !       DYDA(NA) are the first derivatives of Y(X,A) over A(j),
      !       NA is the number of unknown coefficients A(j).
      !
      !   IF YOU CHANGE the approx. function, YOU MUST CHANGE 
      !     properly MA = NA in the operator PARAMETER!
      !
      ! OUTPUT: 
      !    K, lambda_max, P_max with their variances and chi^2 are printed 
      !    and written in the file 'out1.txt' by WRITE (17,...)
      !
      !    Values of 1/lambda(i), P(i), approximation P(i), lambda(i) 
      !      are written in the file 'out2.txt' by WRITE(18,...)
      !
      ! 2011 Feb (VI) for Pm, Lm, K extracting
      ! 2014 Sep (VI)

      ! from Driver for routine MRQMIN
      	PARAMETER (NPT=100, MA=2)
      	DIMENSION X(NPT),Y(NPT),SIG(NPT),A(MA),LISTA(MA),  &
      			COVAR(MA,MA),ALPHA(MA,MA),GUES(MA)
          real*4 Lam (npt), P (npt), sig_P (npt)
      	real*4 gPm, gLm, Pti, aK, bK, KK
      	real*4 P_m0, Lam_m0, sP_m0, sLam_m0
      	integer nLam0, nLam, i
      	character*8 star
      	logical test
      	common /KKK/ aK, bK
      	EXTERNAL funcs

      ! PARAMETERS
      aK = 1.86
      	bK = -0.10

      test = .FALSE.
      !	test = .TRUE.
      	tPm = 3.0
      	tlm = 0.9

      	gPm = 0
      	gLm = 0

      open(unit=15,file='inp.txt',status='old',access='sequential')
      open(unit=17,file='out1.txt',status='old',access='append')
      open(unit=18,file='out2.txt',status='old',access='append')
      !      &  status='new',access='sequential')

      print *, ' Start'
	  
      	write (17,212)
        212 format(/'  Name       lambda_max     P_max', 9x,' chi^2')
      write (18,211)

      ! INPUT

          5 continue
      read (15, 210, end=1000) star, nLam0
        210 format (1x, a8, i4)
      print *, ' OBJECT: ', star, nLam0
      	if (nLam0 .gt. npt) print *, ' ERROR: nlam0 > npt!'
      	if (nLam0 .gt. npt) write(15,*) ' ERROR: nlam0 > npt!'

      	if (nLam0 .le. 2) print *, star,' ERROR: nLam0 <= 2 !'
      	nLam0p = 2
      	if (nLam0 .le. 2) nLam0p = 0
      !	do 1177 jj = 1, nLam0
      ! 1177	  read (15, 211) Lam (i), P (i), sig_P (i)
      ! 1178	if (nLam0 .le. 2) goto 5

      	print *, ' '
      	print *, '     lambda                P         sigma_P'
      do 10 i = 1, nLam0			     
        read (15, 211, end=11) Lam (i), P (i), sig_P (i)
        211   format (1x, 2f6.3, f7.3) 

      	  if (test)  P (i) = tPm * exp ( -(aK * tlm + bK) * &
      	     log(tlm / Lam (i))**2 )

        if (P(i) .gt. gPm) then
          gPm = P(i)
          gLm = Lam(i)
        end if
      	  print *, Lam (i), P (i), sig_P (i)
         10 continue 
         11 continue
      nLam = i-1
      	if (nLam .ne. nLam0) print *, ' ERROR: nLam NE nLam0 !'
      	if (nLam .ne. nLam0) write(17,*) ' ERROR: nLam NE nLam0 !'
      gues (1) = gPm
      gues (2) = glm
      print *, ' '
      	print *, ' guess values: ', gues(1), gues(2)

      ! transition to MRQMIN

      	DO 12 I = 1, nLam
      		X(I) = 1/Lam (i) 
      		Y(I) = P (i)
      		SIG(I) = sig_P (i)
12	CONTINUE
      	MFIT=MA
      	DO 13 I=1,MFIT
      		LISTA(I)=I
13	CONTINUE
      	ALAMDA=-1.0
      	DO 14 I=1,MA
      		A(I)=GUES(I)
14	CONTINUE
      	CALL MRQMIN(X,Y,SIG,nLam,A,MA,LISTA,MFIT,COVAR,ALPHA, & 
      	     MA,CHISQ,funcs,ALAMDA)		
      	K=1
      	ITST=0
          1 continue
       	WRITE(*,'(/1X,A,I2,T18,A,F14.4,T45,A,E9.2)') 'Iteration #',K, & 
       	      'Chi-squared:',CHISQ,'ALAMDA:',ALAMDA
      	K=K+1
      	OCHISQ=CHISQ
      	CALL MRQMIN(X,Y,SIG,nLam,A,MA,LISTA,MFIT,COVAR,ALPHA, &
      	     MA,CHISQ,funcs,ALAMDA)		
      	IF (CHISQ.GT.OCHISQ) THEN
      		ITST=0
      	ELSE IF (ABS(OCHISQ-CHISQ).LT.0.1) THEN
      		ITST=ITST+1
      	ENDIF
      	IF (ITST.LT.2) THEN
      		GOTO 1
      	ENDIF
      	ALAMDA=0.0
      	CALL MRQMIN(X,Y,SIG,nLam,A,MA,LISTA,MFIT,COVAR,ALPHA, & 
      	     MA,CHISQ,funcs,ALAMDA)		

      ! transition back

      	P_m0 = a (1)
      	Lam_m0 = a(2)
      	sP_m0 = sqrt (COVAR(1,1))	
      	sLam_m0 = sqrt (COVAR(2,2))	

      ! OUTPUT

      	print *, ' '
      	print *, '    lambda_max         P_max'
      print *, Lam_m0, P_m0
      print *, sLam_m0, sP_m0
      write (17,220)	star, Lam_m0, sLam_m0, P_m0, sP_m0, &
         CHISQ / (nLam - nLam0p) 
        220	format (1x, A8, 2(2x, F6.3,'+-',F5.3), 1x,F8.2)

      	print *, ' '
      	print *, '                              P/Pmax  '
      	print *, '     lambda          observ          theory' 
      	KK = aK * Lam_m0 + bK
	
      	print *, CHISQ, nLam, nLam0p 
	
      	 write (18,213) star, Lam_m0, sLam_m0, P_m0, sP_m0, &
      	    CHISQ / (nLam - nLam0p) 
        213	format (1x, A8, & 
        3x, 'Lam_max=', F6.3,'+-',F5.3, &
        ';  P_max=', F4.2,'+-',F4.2, & 
        ';  chi^2=',F4.1, // & 
        '  lambda   P_cal!       P_obs        1/lam', & 
        '   P_calc/Pmax    P_obs/Pmax     lam_max/lam')

      do 31 i = 1, nLam
        Pti = P_m0 * exp (-KK * log(Lam_m0/Lam(i))**2)
        print *, Lam(i), P (i)/P_m0, Pti/P_m0, 	&
            (P(i) - Pti)/Pti
      	  write (18,221) Lam(i), Pti, P (i), sig_P(i), 1/Lam(i), &
      	     Pti/P_m0, P (i)/P_m0, sig_P (i)/P_m0, Lam_m0/Lam(i)
        221	  format (1x, F5.2, 4x,F6.3, 1x,F8.3,' +-', F5.3, 3x,F5.2, &
       4x,F6.3, 6x,F6.3, ' +-', F5.3, 6x,F5.2)
         31 continue
      !      write (17,221)
      	write (18,221)

      	print *, ' '
      	goto 5
       1000 stop
      	END PROGRAM SERK_OBS_2

      !--------------------------------------------------------------
      ! FUNCS

      SUBROUTINE funcs (X, A, Y, DYDA, NA)
      DIMENSION A(NA), DYDA(NA)
      	real*4 KK, ak, bK
      	common /KKK/ aK, bK
      	KK = aK * a(2) + bK
      Y = a (1) * exp (-KK * ( log(x * a(2)) )**2) 
      DYDA(1) = Y / a (1)
      DYDA(2) = -Y * ( aK * (log(x * a(2)))**2 + &  
        2 * KK / a(2) * log(x * a(2)) )
      RETURN
      END

      ! REST SOFT

      SUBROUTINE MRQMIN(X,Y,SIG,NDATA,A,MA,LISTA,MFIT, &
        COVAR,ALPHA,NCA,CHISQ,FUNCS,ALAMDA)
      external FUNCS
      PARAMETER (MMAX=20)
      DIMENSION X(NDATA),Y(NDATA),SIG(NDATA),A(MA),LISTA(MA), &
        COVAR(NCA,NCA),ALPHA(NCA,NCA),ATRY(MMAX),BETA(MMAX),DA(MMAX)
      IF(ALAMDA.LT.0.)THEN
        KK=MFIT+1
        DO 12 J=1,MA
          IHIT=0
          DO 11 K=1,MFIT
            IF(LISTA(K).EQ.J)IHIT=IHIT+1
11        CONTINUE
          IF (IHIT.EQ.0) THEN
            LISTA(KK)=J
            KK=KK+1
          ELSE IF (IHIT.GT.1) THEN
            print *, 'Improper permutation in LISTA'
          ENDIF
12      CONTINUE
        IF (KK.NE.(MA+1)) print *, 'Improper permutation in LISTA'
        ALAMDA=0.001
        CALL MRQCOF(X,Y,SIG,NDATA,A,MA,LISTA,MFIT,ALPHA,BETA,NCA,CHISQ,FUNCS)
        OCHISQ=CHISQ
        DO 13 J=1,MA
          ATRY(J)=A(J)
13      CONTINUE
      ENDIF
      DO 15 J=1,MFIT
        DO 14 K=1,MFIT
          COVAR(J,K)=ALPHA(J,K)
14      CONTINUE
        COVAR(J,J)=ALPHA(J,J)*(1.+ALAMDA)
        DA(J)=BETA(J)
15    CONTINUE
      CALL GAUSSJ(COVAR,MFIT,NCA,DA,1,1)
      IF(ALAMDA.EQ.0.)THEN
        CALL COVSRT(COVAR,NCA,MA,LISTA,MFIT)
        RETURN
      ENDIF
      DO 16 J=1,MFIT
        ATRY(LISTA(J))=A(LISTA(J))+DA(J)
16    CONTINUE
      CALL MRQCOF(X,Y,SIG,NDATA,ATRY,MA,LISTA,MFIT,COVAR,DA,NCA,CHISQ,FUNCS)
      IF(CHISQ.LT.OCHISQ)THEN
        ALAMDA=0.1*ALAMDA
        OCHISQ=CHISQ
        DO 18 J=1,MFIT
          DO 17 K=1,MFIT
            ALPHA(J,K)=COVAR(J,K)
17        CONTINUE
          BETA(J)=DA(J)
          A(LISTA(J))=ATRY(LISTA(J))
18      CONTINUE
      ELSE
        ALAMDA=10.*ALAMDA
        CHISQ=OCHISQ
      ENDIF
      RETURN
      END

      SUBROUTINE MRQCOF(X,Y,SIG,NDATA,A,MA,LISTA,MFIT,ALPHA,BETA,NALP, &
          CHISQ,FUNCS)
      external FUNCS
      PARAMETER (MMAX=20)
      DIMENSION X(NDATA),Y(NDATA),SIG(NDATA),ALPHA(NALP,NALP),BETA(MA), &
          DYDA(MMAX),LISTA(MFIT),A(MA)
      DO 12 J=1,MFIT
        DO 11 K=1,J
          ALPHA(J,K)=0.
11      CONTINUE
        BETA(J)=0.
12    CONTINUE
      CHISQ=0.
      DO 15 I=1,NDATA
        CALL FUNCS(X(I),A,YMOD,DYDA,MA)
        SIG2I=1./(SIG(I)*SIG(I))
        DY=Y(I)-YMOD
        DO 14 J=1,MFIT
          WT=DYDA(LISTA(J))*SIG2I
          DO 13 K=1,J
            ALPHA(J,K)=ALPHA(J,K)+WT*DYDA(LISTA(K))
13        CONTINUE
          BETA(J)=BETA(J)+DY*WT
14      CONTINUE
        CHISQ=CHISQ+DY*DY*SIG2I
15    CONTINUE
      DO 17 J=2,MFIT
        DO 16 K=1,J-1
          ALPHA(K,J)=ALPHA(J,K)
16      CONTINUE
17    CONTINUE
      RETURN
      END

      SUBROUTINE COVSRT(COVAR,NCVM,MA,LISTA,MFIT)
      DIMENSION COVAR(NCVM,NCVM),LISTA(MFIT)
      DO 12 J=1,MA-1
        DO 11 I=J+1,MA
          COVAR(I,J)=0.
11      CONTINUE
12    CONTINUE
      DO 14 I=1,MFIT-1
        DO 13 J=I+1,MFIT
          IF(LISTA(J).GT.LISTA(I)) THEN
            COVAR(LISTA(J),LISTA(I))=COVAR(I,J)
          ELSE
            COVAR(LISTA(I),LISTA(J))=COVAR(I,J)
          ENDIF
13      CONTINUE
14    CONTINUE
      SWAP=COVAR(1,1)
      DO 15 J=1,MA
        COVAR(1,J)=COVAR(J,J)
        COVAR(J,J)=0.
15    CONTINUE
      COVAR(LISTA(1),LISTA(1))=SWAP
      DO 16 J=2,MFIT
        COVAR(LISTA(J),LISTA(J))=COVAR(1,J)
16    CONTINUE
      DO 18 J=2,MA
        DO 17 I=1,J-1
          COVAR(I,J)=COVAR(J,I)
17      CONTINUE
18    CONTINUE
      RETURN
      END

      SUBROUTINE GAUSSJ(A,N,NP,B,M,MP)
      PARAMETER (NMAX=50)
      DIMENSION A(NP,NP),B(NP,MP),IPIV(NMAX),INDXR(NMAX),INDXC(NMAX)
      DO 11 J=1,N
        IPIV(J)=0
11    CONTINUE
      DO 22 I=1,N
        BIG=0.
        DO 13 J=1,N
          IF(IPIV(J).NE.1)THEN
            DO 12 K=1,N
              IF (IPIV(K).EQ.0) THEN
                IF (ABS(A(J,K)).GE.BIG)THEN
                  BIG=ABS(A(J,K))
                  IROW=J
                  ICOL=K
                ENDIF
              ELSE IF (IPIV(K).GT.1) THEN
                print *, 'Singular matrix'
              ENDIF
12          CONTINUE
          ENDIF
13      CONTINUE
        IPIV(ICOL)=IPIV(ICOL)+1
        IF (IROW.NE.ICOL) THEN
          DO 14 L=1,N
            DUM=A(IROW,L)
            A(IROW,L)=A(ICOL,L)
            A(ICOL,L)=DUM
14        CONTINUE
          DO 15 L=1,M
            DUM=B(IROW,L)
            B(IROW,L)=B(ICOL,L)
            B(ICOL,L)=DUM
15        CONTINUE
        ENDIF
        INDXR(I)=IROW
        INDXC(I)=ICOL
        IF (A(ICOL,ICOL).EQ.0.) print *, 'Singular matrix.'
        PIVINV=1./A(ICOL,ICOL)
        A(ICOL,ICOL)=1.
        DO 16 L=1,N
          A(ICOL,L)=A(ICOL,L)*PIVINV
16      CONTINUE
        DO 17 L=1,M
          B(ICOL,L)=B(ICOL,L)*PIVINV
17      CONTINUE
        DO 21 LL=1,N
          IF(LL.NE.ICOL)THEN
            DUM=A(LL,ICOL)
            A(LL,ICOL)=0.
            DO 18 L=1,N
              A(LL,L)=A(LL,L)-A(ICOL,L)*DUM
18          CONTINUE
            DO 19 L=1,M
              B(LL,L)=B(LL,L)-B(ICOL,L)*DUM
19          CONTINUE
          ENDIF
21      CONTINUE
22    CONTINUE
      DO 24 L=N,1,-1
        IF(INDXR(L).NE.INDXC(L))THEN
          DO 23 K=1,N
            DUM=A(K,INDXR(L))
            A(K,INDXR(L))=A(K,INDXC(L))
            A(K,INDXC(L))=DUM
23        CONTINUE
        ENDIF
24    CONTINUE
      RETURN
      END

      ! NOT USED

      FUNCTION GASDEV(IDUM)
      DATA ISET/0/
      IF (ISET.EQ.0) THEN
1       V1=2.*RAN1(IDUM)-1.
        V2=2.*RAN1(IDUM)-1.
        R=V1**2+V2**2
        IF(R.GE.1.)GO TO 1
        FAC=SQRT(-2.*LOG(R)/R)
        GSET=V1*FAC
        GASDEV=V2*FAC
        ISET=1
      ELSE
        GASDEV=GSET
        ISET=0
      ENDIF
      RETURN
      END

      FUNCTION RAN1(IDUM)
      DIMENSION R(97)
      PARAMETER (M1=259200,IA1=7141,IC1=54773,RM1=3.8580247E-6)
      PARAMETER (M2=134456,IA2=8121,IC2=28411,RM2=7.4373773E-6)
      PARAMETER (M3=243000,IA3=4561,IC3=51349)
      DATA IFF /0/
      IF (IDUM.LT.0.OR.IFF.EQ.0) THEN
        IFF=1
        IX1=MOD(IC1-IDUM,M1)
        IX1=MOD(IA1*IX1+IC1,M1)
        IX2=MOD(IX1,M2)
        IX1=MOD(IA1*IX1+IC1,M1)
        IX3=MOD(IX1,M3)
        DO 11 J=1,97
          IX1=MOD(IA1*IX1+IC1,M1)
          IX2=MOD(IA2*IX2+IC2,M2)
          R(J)=(FLOAT(IX1)+FLOAT(IX2)*RM2)*RM1
11      CONTINUE
        IDUM=1
      ENDIF
      IX1=MOD(IA1*IX1+IC1,M1)
      IX2=MOD(IA2*IX2+IC2,M2)
      IX3=MOD(IA3*IX3+IC3,M3)
      J=1+(97*IX3)/M3
      IF(J.GT.97.OR.J.LT.1) print *, '!!!'
      RAN1=R(J)
      R(J)=(FLOAT(IX1)+FLOAT(IX2)*RM2)*RM1
      RETURN
      END
