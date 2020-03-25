
function [PDC, AR, popt, SIGMA, S] = stationary_PDC(DATA, Metric, Norm, P_selector, freq, alpha)

%==========================================================================
% [M.F.Pagnotta - March 2020]
%--------------------------------------------------------------------------
% Code to derive the different variants of Partial Directed Coherence (PDC)
%==========================================================================
% INPUT
% - DATA:           DATA.samp:  time series (signals) [nSamp, nChan, nTr]
%                   DATA.Fs:    sampling frequency in Hz (scalar)
%
% - Metric:         'pdc'    - PDC (classic definition)
%                   'gpdc'   - generalized PDC
%                   'ipdc'   - information PDC
%
% - Norm:           'columns' - column-wise (all variants)
%                   'rows'    - row-wise (only PDC)
%
% - P_selector:     P_selector.pmin: minimum model order tested
%                   P_selector.pmax: maximum model order tested
%                   (if pre-selected use: pmin=pmax)
% 
% - freq:           frequencies (vector)
%
% - alpha:          if ~=0: alpha-level for asymptotic statistics
%
%--------------------------------------------------------------------------
% OUTPUT
% - PDC:        connectivity estimates [nChan, nChan, nFreqs]
% - AR:         MVAR coefficients [nChan, nChan, p]
% - popt:       optimal model order
% - SIGMA:      noise covariance matrix (residual error) [nChan, nChan]
% - S:          complex spectral matrix [nChan, nChan, nFreqs]
%==========================================================================

if nargin < 6
    alpha = 0;
end
if or(alpha<0, alpha>1)
    error('problem: significance level')
end


% Information and data:
[nSamp, nChan, nTr] = size(DATA.samp);
nFreqs  = numel(freq);
Fs      = DATA.Fs;
Y       = permute(DATA.samp, [2 1 3]);                                      % [nChan, nSamp, nTr]

% Range for searching optimal model order:
pmin    = P_selector.pmin;
pmax    = P_selector.pmax;



%========== Stationary MVAR fitting =======================================
[~, AR_2d, SIGMA, sbc] = arfit(permute(Y,[2 1 3]), pmin, pmax, 'sbc', 'zero');
popt    = pmin - 1 + find(sbc==min(sbc));
AR_3d   = reshape(AR_2d, [nChan nChan popt]);                               % MVAR model coefficients



%========== PDC COMPUTATION ===============================================
PDC = zeros(nChan, nChan, nFreqs);

%--------------------
%     A(f)
%--------------------
freqAx = freq';
nFreqs = numel(freqAx);
z      = exp(-1i*2*pi*freqAx/Fs);
AR     = AR_3d;
AF     = repmat(eye(nChan), [1 1 nFreqs]);
for oo = 1:popt
    AF = AF + repmat(-AR_3d(:, :, oo), [1 1 nFreqs]) .* repmat(permute(z.^(oo), [3 2 1]), [nChan nChan]);
end
clear oo

%--------------------
%     S(f)
%--------------------
if nargout > 4
    % Cross-spectral density matrix:
    S = zeros(nChan, nChan, nFreqs);
    for nf = 1:nFreqs
        AF_f = squeeze(AF(:,:,nf));
        S(:,:,nf) = AF_f\SIGMA/AF_f';
    end
    clear nf
end



%--------------------
%     PDC(f)
%--------------------
switch Norm
    %----------------------------------------------------------------------
    % [COLUMN-normalized definitions]
    %----------------------------------------------------------------------
    case 'columns'
        switch lower(Metric)
            case {'pdc'}
                PDC = ratio(abs(AF).^2, sum(abs(AF).^2, 1));
            case {'gpdc'}
                SYd    = mdiag(SIGMA);
                invSYd = SYd\eye(nChan);
                sigmaMAT = repmat(diag(invSYd), [1 nChan nFreqs]);
                NUM = sigmaMAT.*(abs(AF).^2);
                DEN = sum(NUM, 1);
                PDC = ratio(NUM, DEN);
            case {'ipdc'}
                SYd      = mdiag(SIGMA);
                invSYd   = SYd\eye(nChan);
                invSYGMA = SIGMA\eye(nChan);
                for ff = 1:1:nFreqs
                    for cc = 1:1:nChan
                        tmp = squeeze(AF(:,cc,ff));
                        a_den(cc) = tmp'*invSYGMA*tmp;
                    end
                    PDC(:,:,ff) = (invSYd*abs(AF(:,:,ff)).^2)./abs(a_den(ones(1,nChan),:));
                end
        end
        
    %----------------------------------------------------------------------
    % [ROW-normalized definition]
    %----------------------------------------------------------------------
    case 'rows'
        switch lower(Metric)
            case {'pdc'}
                PDC = ratio(abs(AF).^2, sum(abs(AF).^2, 2));
            case {'gpdc'}
                error('Row-wise normalization is not available for gPDC')
            case {'ipdc'}
                error('Row-wise normalization is not available for iPDC')
        end
        
end



%========== Asymptotic statistics =========================================
if alpha ~= 0
    % Initialize:
    tmp_data = DATA.samp;
    tmp_data = permute(tmp_data, [1 3 2]);
    tmp_data = reshape(tmp_data, [size(tmp_data,1)*size(tmp_data,2), size(tmp_data,3)]);       % [nSamp*nTr, nChan]
    Z        = zmatrm(tmp_data', popt);
    gamma    = Z*Z';
    clear Z
    
    % Patnaik approximation:
    Patnaik = zeros(nChan, nChan, nFreqs);
    switch lower(Metric)
        case {'pdc'}
            pu_num = eye(nChan);
            pu_den = eye(nChan);
        case {'gpdc'}
            pu_num = mdiag(SIGMA);
            pu_den = mdiag(SIGMA);
        case {'ipdc'}
            pu_num = mdiag(SIGMA);
            pu_den = SIGMA;
    end
    pu_num = pinv(pu_num);
    pu_den = pinv(pu_den);
    
    G1  = single(inv(gamma));
    nn  = size(G1,1);
    PU1 = sqrt(pu_num)*SIGMA*sqrt(pu_num)*nSamp;
    clear gamma
    Omega = single(repmat(PU1,nn,nn));
    ind1  = [1:size(PU1,1):size(Omega,1) size(Omega,1)+1];
    ind2  = [1:size(PU1,1):size(Omega,1) size(Omega,1)+1];
    for j = 1:length(ind2)-1
        for i = 1:length(ind1)-1
            Omega(ind1(i):ind1(i+1)-1,ind2(j):ind2(j+1)-1) = repmat(G1(i,j),size(PU1,1),size(PU1,1)).*Omega(ind1(i):ind1(i+1)-1,ind2(j):ind2(j+1)-1);
        end
    end
    clear G1 j i
    Omega_a = repmat(Omega,2,2);
    clear Omega
    
    for iFreqs = 1:nFreqs
        f = iFreqs/Fs;
        % Matrix C
        Cmat = [];
        Smat = [];
        for r = 1:popt
            divect  = 2*pi*f*r*ones(1,nChan^2);
            cvector = cos(divect);
            svector = -sin(divect);
            Cmat    = [Cmat diag(cvector)];
            Smat    = [Smat diag(svector)];
        end
        Zmat    = zeros(size(Cmat));
        Cjoint  = [Cmat Zmat; Zmat - Smat];
        Ct      = Cjoint*Omega_a*Cjoint';
        clear Cmat Zmat Smat Cjoint
        
        for iu = 1:nChan
            for ju = 1:nChan
                % Eigenvalue computation:
                Co  = [Ct((ju-1)*nChan+iu,(ju-1)*nChan+iu) ...
                       Ct((ju-1)*nChan+iu,(nChan+ju-1)*nChan+iu);
                       Ct((nChan+ju-1)*nChan+iu,(ju-1)*nChan+iu)  ...
                       Ct((nChan+ju-1)*nChan+iu,(nChan+ju-1)*nChan+iu)];
                v   = eig(real(Co));
                Pat = gaminv( (1-alpha), sum(v)^2/(2*v'*v), 2 );
                switch Norm
                    case 'columns'
                        tmp = squeeze(AF(:,ju,iFreqs));
                        dL = tmp'*pu_den*tmp;
                        Patnaik(iu,ju,iFreqs) = Pat/((sum(v)/(v'*v))*(nSamp*abs(dL)));
                    case 'rows'
                        tmp = squeeze(AF(iu,:,iFreqs));
                        dL  = tmp*pu_den*tmp';
                        Patnaik(iu,ju,iFreqs) = Pat/((sum(v)/(v'*v))*(nSamp*abs(dL)));
                end
            end
        end
        clear iu ju
    end
    clear iFreqs
    
    
    % Significant PDC values on frequency scale:
    pdc = PDC;
    clear PDC
    tempThresh = (pdc-Patnaik>0).*pdc+(pdc-Patnaik<=0)*(-1);
    tempThresh(ind2sub(size(tempThresh),find(tempThresh==-1))) = NaN; 
    pdc_th = tempThresh;
    
    PDC.pdc     = pdc;
    PDC.pdc_th  = pdc_th;
    PDC.Patnaik = Patnaik;
end






%==========================================================================
%   Useful functions
%==========================================================================
function A_d = mdiag(A)
A_d = diag(diag(A));

%--------------------------------------------------------------------------
function Z = zmatrm(Y,p)
[K, T] = size(Y);
y1 = [zeros(K*p,1); reshape(flipud(Y),K*T,1)];
Z  =  zeros(K*p,T);
for i=0:T-1
   Z(:,i+1) = flipud(y1(1+K*i:K*i+K*p));
end
