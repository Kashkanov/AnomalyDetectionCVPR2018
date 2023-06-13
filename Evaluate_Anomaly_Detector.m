clc
clear all
close all
 

C3D_CNN_Path='C:/Users/Ermina Tepora/Documents/Karl Files/THESIS/AnomalyDetectionCVPR2018/data/test_features'; % C3D features for videos
Testing_VideoPath='C:/Users/Ermina Tepora/Documents/Karl Files/THESIS/AnomalyDetectionCVPR2018/data/test_videos'; % Path of mp4 videos
AllAnn_Path='C:/Users/Ermina Tepora/Documents/Karl Files/THESIS/AnomalyDetectionCVPR2018/data/test_annotations'; % Path of Temporal Annotations
Model_Score_Folder='C:/Users/Ermina Tepora/Documents/Karl Files/THESIS/AnomalyDetectionCVPR2018/output';  % Path of Pretrained Model score on Testing videos (32 numbers for 32 temporal segments)
Paper_Results='C:/Users/Ermina Tepora/Documents/Karl Files/THESIS/AnomalyDetectionCVPR2018/output';   % Path to save results.
 
All_Videos_scores=dir(Model_Score_Folder);
All_Videos_scores=All_Videos_scores(3:end);
nVideos=length(All_Videos_scores);
frm_counter=1;
All_Detect=zeros(1,1000000);
All_GT=zeros(1,1000000);
zt=0;

for ivideo=1:nVideos
    ivideo
    zt = zt + 1; 
    filename = [AllAnn_Path,'/',All_Videos_scores(ivideo).name(1:end-4),'_ann.txt'];
    fprintf("filename = %s\n", filename);
    data=importdata(filename);
    fprintf("data.data = %s\n", data.textdata{1,1});
    fprintf("zt = %d\n", zt);
    save([AllAnn_Path,'/',All_Videos_scores(ivideo).name(1:end-4),'_ann'], 'data');

    Ann_Path=[AllAnn_Path,'/',All_Videos_scores(ivideo).name(1:end-4),'_ann.mat'];
    load(Ann_Path)
    check=strmatch(All_Videos_scores(ivideo).name(1:end-6),data.textdata{1,1});
    if isempty(check)
         error('????') 
    end

    VideoPath=[Testing_VideoPath,'/', All_Videos_scores(ivideo).name(1:end-4),'.mp4'];
    ScorePath=[Model_Score_Folder,'/', All_Videos_scores(ivideo).name(1:end-4),'.mat'];

  %% Load Video
    
    fprintf('%s\n', VideoPath);
    xyloObj = VideoReader(VideoPath);
  

    Predic_scores=load(ScorePath);
    fps=30;
    Actual_frames=round(xyloObj.Duration*fps);

    Folder_Path=[C3D_CNN_Path,'/',All_Videos_scores(ivideo).name(1:end-4)];
    AllFiles=dir([Folder_Path,'/*.txt']);
    nFileNumbers=length(AllFiles);
    nFrames_C3D=nFileNumbers*16;  % As the features were computed for every 16 frames


%% 32 Shots
    Detection_score_32shots=zeros(1,nFrames_C3D);
    Thirty2_shots= round(linspace(1,length(AllFiles),33));
    fprintf(class(Thirty2_shots(1)));
    Shots_Features=[];
    p_c=0;
    fprintf("%f\n", length(Thirty2_shots))

    for ishots=1:15

        p_c=p_c+1;
        ss=Thirty2_shots(ishots);
        ee=Thirty2_shots(ishots+1)-1;

        if ishots==length(Thirty2_shots)
            ee=Thirty2_shots(ishots+1);
        end
        
        ee = int32(ee);
        ss = int32(ss);
        fprintf("%d\n", ee);

        if ee<ss
            Detection_score_32shots((ss-1)*16+1:(ss-1)*16+1+15)=Predic_scores.predictions(p_c);   
        else
            Detection_score_32shots((ss-1)*16+1:(ee-1)*16+16)=Predic_scores.predictions(p_c);
        end

    end


    Final_score=  [Detection_score_32shots,repmat(Detection_score_32shots(end),[1,Actual_frames-length(Detection_score_32shots)])];
    GT=zeros(1,Actual_frames);

    for ik=1:size(data.data,1)
            st_fr=max(data.data(ik,1),1); 
            end_fr=min(data.data(ik,2),Actual_frames);
            GT(st_fr:end_fr)=1;
    end


    if data.data(1,1)==0.05   % For Normal Videos
        GT=zeros(1,Actual_frames);
    end


    Final_score= ones(1,length(Final_score));
    % subplot(2,1,1); bar(Final_score)
    % subplot(2,1,2); bar(GT)

    All_Detect(frm_counter:frm_counter+length(Final_score)-1)=Final_score;
    All_GT(frm_counter:frm_counter+length(Final_score)-1)=GT;
    frm_counter=frm_counter+length(Final_score);


end


All_Detect=(All_Detect(1:frm_counter-1));
All_GT=All_GT(1:frm_counter-1);
scores=All_Detect;
[so,si] = sort(scores,'descend');
tp=All_GT(si)>0;
fp=All_GT(si)==0;
tp=cumsum(tp);
fp=cumsum(fp);
nrpos=sum(All_GT);
rec=tp/nrpos;
fpr=fp/sum(All_GT==0);

prec=tp./(fp+tp);
AUC1 = trapz(fpr ,rec );
% You can also use the following codes
[X,Y,T,AUC] = perfcurve(All_GT,All_Detect,1);

plot(X,Y,'LineWidth',3.5);
hold on;
AUC_All=[AUC_All;AUC]
clear X  Y

AUC_All*100
grid on
 
