% Select the BP pretraining epoch on validation scenarios only.
clear; clc;
fprintf('===== BP validation selection (normalized ISE) =====\n');

data_dir=fileparts(mfilename('fullpath'));
load(fullfile(data_dir,'bp_weight_candidates.mat'),'candidate_w1','candidate_w2','epochs');

Ts=0.01;
validation=build_validation(Ts);
scores=zeros(epochs,1);
ise_scores=zeros(epochs,1);
du_scores=zeros(epochs,1);

for ep=1:epochs
    [ise_scores(ep),du_scores(ep)]=evaluate_candidate(candidate_w1{ep},candidate_w2{ep},validation,Ts);
    scores(ep)=ise_scores(ep)+0.01*du_scores(ep);
    fprintf('  Epoch %2d: normalized ISE=%.6f, normalized du=%.6f, score=%.6f\n', ...
        ep,ise_scores(ep),du_scores(ep),scores(ep));
end

[best_score,best_epoch]=min(scores);
w1=candidate_w1{best_epoch};
w2=candidate_w2{best_epoch};
save(fullfile(data_dir,'bp_mimo_pretrained.mat'),'w1','w2');
save(fullfile(data_dir,'bp_validation_selection.mat'), ...
    'scores','ise_scores','du_scores','best_score','best_epoch');
fprintf('Selected epoch %d, score=%.6f\n',best_epoch,best_score);
fprintf('  -> %s\n',fullfile(data_dir,'bp_mimo_pretrained.mat'));

function scenarios=build_validation(Ts)
    n=1200;t=(0:n-1)*Ts;
    scenarios={};
    scenarios{end+1}=scenario('val_step_r1_085',0.85*ones(1,n),zeros(1,n));
    scenarios{end+1}=scenario('val_step_r2_085',zeros(1,n),0.85*ones(1,n));
    scenarios{end+1}=scenario('val_ramp_r1_110',min(1.10,t/5),zeros(1,n));
    scenarios{end+1}=scenario('val_ramp_r2_110',zeros(1,n),min(1.10,t/5));
    scenarios{end+1}=scenario('val_sine_r1_005',0.40*sin(2*pi*0.05*t),zeros(1,n));
    scenarios{end+1}=scenario('val_sine_r2_007',zeros(1,n),0.40*sin(2*pi*0.07*t));
    scenarios{end+1}=scenario('val_dual_065',0.65*ones(1,n),0.65*ones(1,n));
end

function sc=scenario(name,r1,r2)
    sc=struct('name',name,'r1',r1,'r2',r2,'N_sim',length(r1));
end

function [total_ise,total_du]=evaluate_candidate(w1,w2,scenarios,Ts)
    total_ise=0;total_du=0;
    for i=1:numel(scenarios)
        out=simulate_frozen(w1,w2,scenarios{i},Ts);
        ref_scale1=max(rms(out.r1),0.5);
        ref_scale2=max(rms(out.r2),0.5);
        total_ise=total_ise+2.0*Ts*sum(out.e1.^2)/(out.N*Ts*ref_scale1^2);
        total_ise=total_ise+Ts*sum(out.e2.^2)/(out.N*Ts*ref_scale2^2);
        total_du=total_du+mean((out.du1/8).^2)+mean((out.du2/8).^2);
    end
end

function out=simulate_frozen(w1,w2,sc,Ts)
    Kp1_base=7.00;Ki1_base=3.00;Kp2_base=0.80;Ki2_base=6.50;
    K_delta=[18.0,14.0,0.75,20.0];
    K_lower=[0.50,0.05,0.05,0.05];K_upper=[30.0,20.0,2.00,30.0];
    Kp2_sign=-1;Ki2_sign=-1;du_max=8.0;u_sat=30;
    beta1=1.00;beta2=0.20;e_freeze=0.10;
    A=[-0.301,-0.050;0.325,-0.375];
    B=[0.079,0.099;0.133,0.302];
    C=[1,0;0.038,0.67];D=[0,0;0.19,-1.31];
    G_ss=D-C*(A\B);K_ff=inv(G_ss);
    alpha_ff=0.20;D_cancel=-D(2,1)/D(2,2);
    tau_ref=0.12;alpha_ref=Ts/(Ts+tau_ref);
    tau_lead=0.8;dr_max=0.25;K_gain_boost=1.5;e1_thresh_gs=0.3;

    clear plant_engine_mimo;
    N=sc.N_sim;
    y1_1=0;y2_1=0;u1_1=0;u1_2=0;u1_3=0;u2_1=0;u2_2=0;u2_3=0;
    u1_pid=0;u2_pid=0;u1_ff=0;r1_1=0;r2_1=0;r1_filt=0;
    e1_sp_1=0;e2_sp_1=0;
    e1=zeros(1,N);e2=zeros(1,N);du1_arr=zeros(1,N);du2_arr=zeros(1,N);

    for k=1:N
        r1_raw=sc.r1(k);r2_raw=sc.r2(k);
        r1_filt=alpha_ref*r1_raw+(1-alpha_ref)*r1_filt;
        r1_k=r1_filt;r2_k=r2_raw;
        e1_filt=r1_k-y1_1;e2_filt=r2_k-y2_1;
        I1=[r1_k,y1_1,e1_filt,r1_k-r1_1,r2_k,y2_1,e2_filt,r2_k-r2_1,1];
        O2=tanh(I1*w1');O3=tanh((w2*O2')');
        gain_raw=[Kp1_base,Ki1_base,Kp2_base,Ki2_base]+K_delta.*O3;
        gain_abs=max(K_lower,min(K_upper,gain_raw));
        Kp1=gain_abs(1);Ki1=gain_abs(2);
        Kp2=Kp2_sign*gain_abs(3);Ki2=Ki2_sign*gain_abs(4);
        Kp1_eff=Kp1*(1+K_gain_boost*exp(-abs(e1_filt)/e1_thresh_gs));

        e1_sp=beta1*r1_k-y1_1;e2_sp=beta2*r2_k-y2_1;
        du1=Kp1_eff*(e1_sp-e1_sp_1);
        if abs(e1_filt)<=e_freeze,du1=du1+Ki1*Ts*e1_filt;end
        du2=Kp2*(e2_sp-e2_sp_1)+Ki2*Ts*e2_filt;
        du1=max(-du_max,min(du_max,du1));du2=max(-du_max,min(du_max,du2));
        u1_pid=u1_pid+du1;u2_pid=u2_pid+du2;

        dr1=(r1_k-r1_1)/Ts;dr1_clip=max(-dr_max,min(dr_max,dr1));
        r1_lead=r1_k+tau_lead*dr1_clip;
        u1_ff_raw=K_ff(1,1)*r1_lead+K_ff(1,2)*r2_k;
        u1_ff=alpha_ff*u1_ff_raw+(1-alpha_ff)*u1_ff;
        u1_k=max(-u_sat,min(u_sat,u1_ff+u1_pid));
        u2_k=max(-u_sat,min(u_sat,u2_pid+D_cancel*u1_k));

        [y1_k,y2_k]=plant_engine_mimo(y1_1,y2_1,u1_1,u1_2,u1_3,u2_1,u2_2,u2_3);
        e1(k)=r1_raw-y1_k;e2(k)=r2_raw-y2_k;
        du1_arr(k)=du1;du2_arr(k)=du2;
        y1_1=y1_k;y2_1=y2_k;
        u1_3=u1_2;u1_2=u1_1;u1_1=u1_k;u2_3=u2_2;u2_2=u2_1;u2_1=u2_k;
        r1_1=r1_k;r2_1=r2_k;e1_sp_1=e1_sp;e2_sp_1=e2_sp;
    end
    out=struct('r1',sc.r1,'r2',sc.r2,'e1',e1,'e2',e2, ...
        'du1',du1_arr,'du2',du2_arr,'N',N);
end
