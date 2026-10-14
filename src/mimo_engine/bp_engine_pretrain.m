% bp_engine_pretrain.m
% BP离线预训练 — 混合信号: 7阶段覆盖全部7个测试场景类型
% 每epoch: step r1|step r2|r1 ramp|r2 ramp|r1 sine|r2 sine|dual step
% 频率加密: 5种正弦频率循环(排除测试频率0.06/0.08Hz)
% 输入加频率特征: Δr₁ RMS + Δr₂ RMS (11-8-4)

clear; clc;
fprintf('===== BP Pretrain (7-phase mixed signal, 96 epochs, 11-8-4) =====\n');

Ts=0.01; N_sim=2800;
A=[-0.301,-0.050;0.325,-0.375];
B_mat=[0.079,0.099;0.133,0.302];
C_mat=[1,0;0.038,0.67];
D_mat=[0,0;0.19,-1.31];

%% 参数 (与run_engine_test.m一致)
IN=9; H_bp=8; Out_bp=4;
Kp1_base=7.00; Ki1_base=3.00;
Kp2_base=0.80; Ki2_base=6.50;
K_delta=[18.0,14.0,0.75,20.0];
K_lower=[0.50,0.05,0.05,0.05];
K_upper=[30.0,20.0,2.00,30.0];
Kp2_sign=-1; Ki2_sign=-1;
du_max=8.0; u_sat=30;
beta1=1.00; beta2=0.20;
e_freeze=0.10;

G_ss = D_mat - C_mat * (A \ B_mat);
K_ff = inv(G_ss);
alpha_ff = 0.20;
D_cancel = -D_mat(2,1)/D_mat(2,2);
K_boost = 0.0;
tau_ref = 0.12;  alpha_ref = Ts/(Ts+tau_ref);
tau_lead = 0.8;  dr_max = 0.25;
K_gain_boost = 1.5;  e1_thresh_gs = 0.3;
rate_pt = 0.05; jac_cap=2.0; dead_zone=0.001;
channel_loss_weight=[2.0,1.0];

%% BP初始化
rng(42);
w1 = 0.2 * randn(H_bp, IN);
w2 = 0.15 * randn(Out_bp, H_bp);
fprintf('BP: (9-10-4) H_bp=%d epochs=%d\n', H_bp, epochs);

%% 阶跃幅值序列 (12 amplitudes, 96 epochs) — 已排除测试集幅值0.5和1.0
step_amps = repmat([0.3,0.55,0.75,0.9,1.15,1.35,1.5,0.45,0.65,0.85,1.05,1.25],1,8);
epochs = length(step_amps);
%% 正弦频率序列 (5频率循环，排除测试集0.06/0.08 Hz)
sine_freqs_r1 = repmat([0.04, 0.05, 0.07, 0.09, 0.10, 0.04, 0.05, 0.07, 0.09, 0.10, 0.04, 0.05], 1, 8);
sine_freqs_r2 = repmat([0.10, 0.09, 0.07, 0.05, 0.04, 0.10, 0.09, 0.07, 0.05, 0.04, 0.10, 0.09], 1, 8);
fprintf('预训练: %d epochs x %d 步, 7阶段混合信号\n', epochs, N_sim);
fprintf('阶跃幅值: %s\n', mat2str(step_amps(1:12)));
fprintf('正弦频率r1: %s\n', mat2str(unique(sine_freqs_r1)));
fprintf('正弦频率r2: %s\n', mat2str(unique(sine_freqs_r2)));
loss_log = {};
snap_y1 = cell(1, epochs); snap_y2 = cell(1, epochs);
snap_r1 = cell(1, epochs); snap_r2 = cell(1, epochs);
snap_labels = cell(1, epochs);
candidate_w1 = cell(1, epochs); candidate_w2 = cell(1, epochs);

for ep = 1:epochs
    amp = step_amps(ep);
    freq_r1 = sine_freqs_r1(ep);
    freq_r2 = sine_freqs_r2(ep);
    fprintf('\n--- Epoch %d/%d (amp=%.1f, f_r1=%.2f, f_r2=%.2f) ---\n', ep, epochs, amp, freq_r1, freq_r2);
    clear plant_engine_mimo;
    rbf_identifier_mimo(0,0,0,0,'freeze');

    y1_1=0;y2_1=0;u1_1=0;u1_2=0;u1_3=0;u2_1=0;u2_2=0;u2_3=0;
    u1_pid=0;u2_pid=0;u1_ff=0;r1_1=0;r2_1=0;r1_filt=0;
    e1_sp_1=0;e2_sp_1=0;

    st_has=false;st_O2=zeros(1,H_bp);st_I1=zeros(1,IN);
    st_dO3=zeros(1,Out_bp);st_dO2=zeros(1,H_bp);st_ep=zeros(1,4);
    w1_b=w1;w1_b2=w1;w2_b=w2;w2_b2=w2;
    mae_sum1=0;mae_sum2=0;

    y1_arr=zeros(1,N_sim); y2_arr=zeros(1,N_sim);
    r1_arr=zeros(1,N_sim); r2_arr=zeros(1,N_sim);

    for k = 1:N_sim
        %% ===== 7阶段混合参考信号 =====
        phase = k / N_sim;
        if phase < 1/7
            % 1: Step r1 Only
            r1_raw = amp;  r2_raw = 0;
        elseif phase < 2/7
            % 2: Step r2 Only
            r1_raw = 0;  r2_raw = amp * 0.7;
        elseif phase < 3/7
            % 3: r1 Ramp
            frac = (phase - 2/7) / (1/7);
            r1_raw = amp * frac;  r2_raw = 0;
        elseif phase < 4/7
            % 4: r2 Ramp
            frac = (phase - 3/7) / (1/7);
            r1_raw = 0;  r2_raw = amp * 0.7 * frac;
        elseif phase < 5/7
            % 5: r1 Sine
            r1_raw = amp * 0.5 * sin(2*pi*freq_r1*k*Ts);  r2_raw = 0;
        elseif phase < 6/7
            % 6: r2 Sine
            r1_raw = 0;  r2_raw = amp * 0.5 * sin(2*pi*freq_r2*k*Ts);
        else
            % 7: Dual Step
            r1_raw = amp * 0.5;  r2_raw = amp * 0.5;
        end

        %% ===== 控制律 (与run_engine_test.m一致) =====
        r1_filt = alpha_ref*r1_raw + (1-alpha_ref)*r1_filt;
        r1_k=r1_filt; r2_k=r2_raw;
        e1_filt=r1_k-y1_1; e2_filt=r2_k-y2_1;

        I1=[r1_k,y1_1,e1_filt,r1_k-r1_1,r2_k,y2_1,e2_filt,r2_k-r2_1,1];
        I2=I1*w1'; O2=tanh(I2); I3=w2*O2'; I3_t=I3';
        O3=tanh(I3_t);
        gain_raw=[Kp1_base,Ki1_base,Kp2_base,Ki2_base]+K_delta.*O3;
        gain_abs=max(K_lower,min(K_upper,gain_raw));

        Kp1=gain_abs(1); Ki1=gain_abs(2);
        Kp2=Kp2_sign*gain_abs(3); Ki2=Ki2_sign*gain_abs(4);
        Kp1_eff = Kp1 * (1 + K_gain_boost * exp(-abs(e1_filt)/e1_thresh_gs));

        e1_sp=beta1*r1_k-y1_1; e2_sp=beta2*r2_k-y2_1;
        de1_sp=e1_sp-e1_sp_1; de2_sp=e2_sp-e2_sp_1;
        du1=Kp1_eff*de1_sp;
        if abs(e1_filt)<=e_freeze, du1=du1+Ki1*Ts*e1_filt; end
        du2=Kp2*de2_sp+Ki2*Ts*e2_filt;
        du1=max(-du_max,min(du_max,du1)); du2=max(-du_max,min(du_max,du2));
        u1_pid=u1_pid+du1; u2_pid=u2_pid+du2;

        dr1 = (r1_k - r1_1)/Ts;
        dr1_clip = max(-dr_max, min(dr_max, dr1));
        r1_lead = r1_k + tau_lead * dr1_clip;
        u1_ff_raw = K_ff(1,1)*r1_lead + K_ff(1,2)*r2_k;
        u1_ff = alpha_ff*u1_ff_raw + (1-alpha_ff)*u1_ff;

        u1_k=max(-u_sat,min(u_sat,u1_ff+u1_pid));
        u2_k=max(-u_sat,min(u_sat,u2_pid+D_cancel*u1_k));

        [y1_k,y2_k]=plant_engine_mimo(y1_1,y2_1,u1_1,u1_2,u1_3,u2_1,u2_2,u2_3);
        y1_arr(k)=y1_k; y2_arr(k)=y2_k;
        r1_arr(k)=r1_raw; r2_arr(k)=r2_raw;
        e1_k2=r1_k-y1_k; e2_k2=r2_k-y2_k;
        mae_sum1=mae_sum1+abs(e1_k2); mae_sum2=mae_sum2+abs(e2_k2);

        %% ===== BP在线学习 =====
        [J_rbf,~]=rbf_identifier_mimo(u1_1,u2_1,y1_k,y2_k,false);

        if st_has && max(abs(e1_k2),abs(e2_k2))>=dead_zone
            J11=max(-jac_cap,min(jac_cap,J_rbf(1,1))); J12=max(-jac_cap,min(jac_cap,J_rbf(1,2)));
            J21=max(-jac_cap,min(jac_cap,J_rbf(2,1))); J22=max(-jac_cap,min(jac_cap,J_rbf(2,2)));
            e1_c=max(-1,min(1,e1_k2)); e2_c=max(-1,min(1,e2_k2));
            dE_du1=channel_loss_weight(1)*e1_c*J11+channel_loss_weight(2)*e2_c*J21;
            dE_du2=channel_loss_weight(1)*e1_c*J12+channel_loss_weight(2)*e2_c*J22;

            delta3=zeros(1,Out_bp);
            delta3(1)=dE_du1*K_delta(1)*st_ep(1)*st_dO3(1);
            delta3(2)=dE_du1*K_delta(2)*Ts*st_ep(2)*st_dO3(2);
            delta3(3)=dE_du2*Kp2_sign*K_delta(3)*st_ep(3)*st_dO3(3);
            delta3(4)=dE_du2*Ki2_sign*K_delta(4)*Ts*st_ep(4)*st_dO3(4);

            d_w2=zeros(Out_bp,H_bp);
            for l=1:Out_bp,for i=1:H_bp,d_w2(l,i)=rate_pt*delta3(l)*st_O2(i);end;end
            w2_new=w2_b+d_w2+rate_pt*0.3*(w2_b-w2_b2);
            a_back=delta3*w2_b; delta2=st_dO2.*a_back;
            d_w1=rate_pt*delta2'*st_I1;
            w1_new=w1_b+d_w1+rate_pt*0.3*(w1_b-w1_b2);
            w1_b2=w1_b; w1_b=w1_new; w2_b2=w2_b; w2_b=w2_new;
            w1=w1_new; w2=w2_new;
        end
        st_has=true; st_ep=[de1_sp,r1_raw-y1_1,de2_sp,r2_raw-y2_1]; st_O2=O2; st_I1=I1;
        active_bound=(gain_raw>K_lower)&(gain_raw<K_upper);
        st_dO3=(1-O3.^2).*active_bound; st_dO2=1-tanh(I2).^2;

        y1_1=y1_k;y2_1=y2_k; u1_3=u1_2;u1_2=u1_1;u1_1=u1_k; u2_3=u2_2;u2_2=u2_1;u2_1=u2_k;
        r1_1=r1_k; r2_1=r2_k;
        e1_sp_1=e1_sp; e2_sp_1=e2_sp;
    end
    fprintf('  MAE1=%.4f  MAE2=%.4f\n', mae_sum1/N_sim, mae_sum2/N_sim);
    label = sprintf('amp=%.1f: step1|step2|ramp1|ramp2|sin1|sin2|dual', amp);
    loss_log{end+1} = struct('epoch', ep, 'amp', amp, ...
        'mae1', mae_sum1/N_sim, 'mae2', mae_sum2/N_sim, 'label', label);
    snap_y1{ep} = y1_arr; snap_y2{ep} = y2_arr;
    snap_r1{ep} = r1_arr; snap_r2{ep} = r2_arr;
    snap_labels{ep} = label;
    candidate_w1{ep} = w1; candidate_w2{ep} = w2;
end

data_dir = fileparts(mfilename('fullpath'));
save(fullfile(data_dir,'bp_mimo_pretrained.mat'),'w1','w2');
fprintf('\n  -> %s\n', fullfile(data_dir,'bp_mimo_pretrained.mat'));

save(fullfile(data_dir,'figdata_bp.mat'),'loss_log','snap_y1','snap_y2','snap_r1','snap_r2','snap_labels','epochs','N_sim');
fprintf('  -> %s\n', fullfile(data_dir,'figdata_bp.mat'));
save(fullfile(data_dir,'bp_weight_candidates.mat'),'candidate_w1','candidate_w2','epochs');
fprintf('  -> %s\n', fullfile(data_dir,'bp_weight_candidates.mat'));
