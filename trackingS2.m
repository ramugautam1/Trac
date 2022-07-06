%% step2: calculating correlation and tracking
clear
load('C:\my3D_matlab\colormap.mat','map')
%source_data=niftiread(strcat('F:\Mo\gt\Myo-cherry and Aju-GFP 2020.12.21Sample\source\','sqh-cherry_jub-gfp 18A-02.nii'));
I3dw=[512, 280, 15];% the shape of source data. remember to change if source data change. size(source_data);
disp('start')
padding=[20, 20, 2]; % the padding on  segmentation for 'extended search', see paper
time=clock;
% folder = 'TrackingEcad2020'; % the folder stores the segmentation data for each time point
folder = 'Recombined_18A-09';
trackbackT=2; % the program support multiple trace back time point.

% ---IF new tracking task, uncomment this part
% filename = strcat('D:\NEW\',folder,'\TrackingID',num2str(time),'.xls'); % the excel file name to write the tracking result

filename = strcat('D:\NEW\',folder,'\TrackingID',num2str(time),'.xls'); % the excel file name to write the tracking result

xlswrite(filename,cellstr('time'),2, 'A1'); % write titles to the excel
xlswrite(filename,cellstr('old'),2, 'B1');
xlswrite(filename,cellstr('new'),2, 'C1');
xlswrite(filename,cellstr('split'),2, 'D1');
xlswrite(filename,cellstr('fusion'),2, 'E1');


% ---IF continue tracking uncomment this part
% filename = strcat('D:\NEW\',folder,'\TrackingID2021              2              2             14             25         14.621.xls');
% load(strcat('D:\NEW\',folder,'\','xlswriter1','.mat'),'xlswriter1');
% load(strcat('D:\NEW\',folder,'\','xlswriter2','.mat'),'xlswriter2');
% load(strcat('D:\NEW\',folder,'\','xlswriter3','.mat'),'xlswriter3');
% load(strcat('D:\NEW\',folder,'\','xlswriter4','.mat'),'xlswriter4');
% load(strcat('D:\NEW\',folder,'\','xlswriter5','.mat'),'xlswriter5');
% load(strcat('D:\NEW\',folder,'\','xlswriter6','.mat'),'xlswriter6');
% load(strcat('D:\NEW\',folder,'\','xlswriter7','.mat'),'xlswriter7');
% load(strcat('D:\NEW\',folder,'\','xlswriter8','.mat'),'xlswriter8');
% load(strcat('D:\NEW\',folder,'\','xlswriter9','.mat'),'xlswriter9');
% load(strcat('D:\NEW\',folder,'\','xlswriter10','.mat'),'xlswriter10');

depth=64; % the deep features to take in correlation calculation
initialpoint=1; % the very first time point of all samples
startpoint=1; % the time point to start tracking
endpoint=41; % the time point to stop tracking

spatial_extend_matrix=zeros(10,10,3,depth); % the weight decay of 'extended search' (not used right now in correlation calculation)
for i1=1:10
    for i2=1:10
        for i3=1:3
            spatial_extend_matrix(i1,i2,i3,:)=exp(((i1-5)+(i2-5)+(i3-2))/20);
        end
    end
end

for time=startpoint:endpoint
    tic;
    disp(strcat('time point: ', num2str(time)))
    t1=num2str(time);
    t2=num2str(time+1);
    xlswriter1(1,(time)*2-1)=cellstr(t1);%add time point in excel
    xlswriter1(1,(time)*2)=cellstr(t2);
    xlswriter3(1,(time)*2)=cellstr(t2);
    xlswriter4(1,(time)*2)=cellstr(t2);
    xlswriter5(1,(time)*2)=cellstr(t2);
    xlswriter6(1,(time)*2)=cellstr(t2);
    xlswriter7(1,(time)*2)=cellstr(t2);
    xlswriter8(1,(time)*2)=cellstr(t2);
    xlswriter9(1,(time)*2)=cellstr(t2);
    xlswriter10(1,(time)*2)=cellstr(t2);
    xlswriter11(1,(time)*2)=cellstr(t2);
    xlswriter12(1,(time)*2)=cellstr(t2);
    addr1=strcat('D:\NEW\',folder,'\',t1,'\');
    addr2=strcat('D:\NEW\',folder,'\',t2,'\');
    Files1=dir(strcat(addr1,'*.nii'));
    Files2=dir(strcat(addr2,'*.nii'));

    if time-initialpoint<trackbackT %calculating correlation for start time points (e.g. time=2)
        for i1=1:time-initialpoint+1
            disp(addr2)
            Fullsize_2 = niftiread(strcat(addr2,'Fullsize_label_',t2,'.nii')); %read segmentation result (class)
            Fullsize_regression_2 = niftiread(strcat(addr2,'Weights_',t2,'.nii')); %read segmentation result (weight)
            if i1==time-initialpoint+1
                disp(addr1)
                Fullsize_1 = niftiread(strcat(addr1,'Fullsize_label_',t1,'.nii'));
                Fullsize_regression_1 = niftiread(strcat(addr1,'Weights_',t1,'.nii'));
            else
                Fullsize_1 = niftiread(strcat(addr1,'Fullsize_2_aftertracking_',t1,'.nii'));
                Fullsize_regression_1 = niftiread(strcat(addr1,'Weights_',t1,'.nii'));
            end
            % ---correlation calculation, uncomment if tracking a sample
            % ---the first time (it will comsume hours, be patient)
            correlation20210408(Fullsize_1,Fullsize_2,Fullsize_regression_1,Fullsize_regression_2,t2,i1,spatial_extend_matrix,addr2,padding);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
    else
        for i1=1:trackbackT %calculating correlation for later time points (e.g. time >2)
            Fullsize_2 = niftiread(strcat(addr2,'Fullsize_label_',t2,'.nii'));
            Fullsize_regression_2 = niftiread(strcat(addr2,'Weights_',t2,'.nii'));
            Fullsize_1 = niftiread(strcat(addr1,'Fullsize_2_aftertracking_',t1,'.nii'));
            Fullsize_regression_1 = niftiread(strcat(addr1,'Weights_',t1,'.nii'));
            correlation20210408(Fullsize_1,Fullsize_2,Fullsize_regression_1,Fullsize_regression_2,t2,i1,spatial_extend_matrix,addr2, padding);
        end
    end


    % clear unnecessary variables
clear Fullsize_1 Fullsize_regression_1 Fullsize_2 Fullsize_regression_2  Fullsize_1_padding Fullsize_2_padding ...
        Fullsize_regression_1_padding Fullsize_regression_2_padding Fullsize_1_label Fullsize_2_label

    % plot tracking
    t1=num2str(time);
    t2=num2str(time+1);

    % read the correlation calculation results
    correlation_map_padding_show1 = niftiread(strcat('D:\NEW\',folder,'\',t2,'\correlation_map_padding_show_traceback1_',t2,'.nii'));
    correlation_map_padding_hide1 = niftiread(strcat('D:\NEW\',folder,'\',t2,'\','correlation_map_padding_show_traceback1','_',t2,'.nii'));

    if time-initialpoint<trackbackT && time>initialpoint
        for i1=1:time-initialpoint
            Registration1 = niftiread(strcat('D:\NEW\',folder,'\',t1,'\Registration2_tracking_',t1,'.nii'));
            correlation_map_padding_show1_2 = niftiread(strcat('D:\NEW\',folder,'\',t2,'\','correlation_map_padding_show_traceback',num2str(i1),'_',t2,'.nii'));
            correlation_map_padding_hide1_2 = niftiread(strcat('D:\NEW\',folder,'\',t2,'\','correlation_map_padding_hide_traceback',num2str(i1),'_',t2,'.nii'));
            for i2=1:I3dw(1)+padding(1)*2 % sign labels based on correlation
                for i3=1:I3dw(2)+padding(2)*2
                    for i4=1:I3dw(3)+padding(3)*2
                        if correlation_map_padding_hide1(i2,i3,i4)<correlation_map_padding_hide1_2(i2,i3,i4) && correlation_map_padding_show1_2(i2,i3,i4)~=0
                            correlation_map_padding_show1(i2,i3,i4)=correlation_map_padding_show1_2(i2,i3,i4); %???
                        end
                    end
                end
            end
        end
    elseif time-initialpoint >=trackbackT && time>initialpoint
        for i1=2:trackbackT
            Registration1 = niftiread(strcat('D:\NEW\',folder,'\',t1,'\Registration2_tracking_',t1,'.nii'));
            correlation_map_padding_show1_2 = niftiread(strcat('D:\NEW\',folder,'\',t2,'\','correlation_map_padding_show_traceback',num2str(i1),'_',t2,'.nii'));
            correlation_map_padding_hide1_2 = niftiread(strcat('D:\NEW\',folder,'\',t2,'\','correlation_map_padding_hide_traceback',num2str(i1),'_',t2,'.nii'));
            for i2=1:I3dw(1)+padding(1)*2
                for i3=1:I3dw(2)+padding(2)*2
                    for i4=1:I3dw(3)+padding(3)*2
                        if correlation_map_padding_hide1(i2,i3,i4)<correlation_map_padding_hide1_2(i2,i3,i4) && correlation_map_padding_show1_2(i2,i3,i4)~=0
                            correlation_map_padding_show1(i2,i3,i4)=correlation_map_padding_show1_2(i2,i3,i4);
                        end
                    end
                end
            end
        end
    else
        Registration1 = niftiread(strcat('D:\NEW\',folder,'\',t1,'\Registration_',t1,'.nii'));
    end

    % read the segmentation
    Fullsize_2 = logical(niftiread(strcat('D:\NEW\',folder,'\',t2,'\Fullsize_',t2,'.nii')));
    Fullsize_2_2= zeros(size(Fullsize_2));
    % crop the expanded sample to its original size
    correlation_map_padding_show2 =correlation_map_padding_show1(21:end-20,21:end-20,3:end-2);
    Fullsize_2_mark=correlation_map_padding_show2;
    if time>initialpoint
        correlation_map_padding_show2_2 =correlation_map_padding_show1_2(21:end-20,21:end-20,3:end-2);
        Fullsize_1 = correlation_map_padding_show2_2;
        Fullsize_1(Fullsize_1==0)=nan;
        % if not initial time point, read the fusion data of last time
        % point
        detector_fusion_old=load(strcat('D:\NEW\',folder,'\',t1,'\fusion_tracking_',t1,'.mat'),'detector3_fusion');
        for i1=2:2:size(detector_fusion_old.detector3_fusion,1)
            detector_fusion_old.detector3_fusion(i1,:)=0;
        end
    end
    Fullsize_2_mark(Fullsize_2==0)=0;

    clear correlation_map_padding_show1 correlation_map_padding_show1_2 correlation_map_padding_hide1 correlation_map_padding_hide1_2

%---draw figures starts, uncomment if needed
% correlation_map_padding_show2(Fullsize_2==0)=0;
%     stack_after_BW=logical(correlation_map_padding_show2);
%     stats = regionprops3(stack_after_BW,'BoundingBox','VoxelList','ConvexHull');
%     disp('Draw overlap figure:')
%
%     correlation_map_padding_show2(correlation_map_padding_show2==0)=nan;
%     [X,Y,Z] = ndgrid(1:size(correlation_map_padding_show2,1), 1:size(correlation_map_padding_show2,2), 1:size(correlation_map_padding_show2,3));
%     pointsize = 5;
%     h1=figure;
%     scatter3(X(:), Y(:), Z(:), pointsize, correlation_map_padding_show2(:),'filled');
%     colormap(map);
%     % zlim([0 100]);
%     colorbar;
%     xlim([0 280]);
%     ylim([0 512]);
%     zlim([0 13]);
%     hold on
%
%     grid on
%
%     for i=1:height(stats)
%
%         b=stats.VoxelList{i,1};
%         %     for j=length(stats.ConvexHull{i,1})
%         %         plot3(a(:,2),a(:,1),a(:,3),'r');
%         k = boundary(b);
%         value=correlation_map_padding_show2(b(end,2),b(end,1),b(end,3));
%
%         hold on
%         trisurf(k,b(:,2),b(:,1),b(:,3),'Facecolor',map(value,1:3),'FaceAlpha',0.1,'Edgecolor',map(value,1:3),'EdgeAlpha',0.1)
%
%
%         %text(b(end,2),b(end,1),b(end,3), num2str(value), 'Rotation',+15)
%     end
%
%
%     hold off
%     savefig(h1,strcat(addr2,t2,'_corroverlap.fig'));
%     close(h1);
%draw figures ends

    % get the object characterestics
    Fullsize_2_mark_BW=Fullsize_2_mark;
    Fullsize_2_mark_BW(Fullsize_2_mark_BW>0)=1;
    Fullsize_2_mark_BW=logical(Fullsize_2_mark_BW);
    [Fullsize_2_mark_label,orgnum]=bwlabeln(Fullsize_2_mark);
    stats1 = regionprops3(Fullsize_2,'BoundingBox','VoxelList','ConvexHull','Centroid');
    sizelist=zeros;
    stats2=table;
    for i1=1:size(stats1,1)%sort the order in terms of size
        sizelist(i1,1)=size(stats1.VoxelList{i1,1},1);
    end
    [sizelistB,sizelistIndex] = sort(sizelist,'descend');
    for i1=1:size(stats1,1)
        stats2.Centroid(i1,1:3)=[stats1.Centroid(sizelistIndex(i1,1),1) stats1.Centroid(sizelistIndex(i1,1),2) stats1.Centroid(sizelistIndex(i1,1),3)];
        stats2.BoundingBox(i1,1:3)=[stats1.BoundingBox(sizelistIndex(i1,1),1) stats1.BoundingBox(sizelistIndex(i1,1),2) stats1.BoundingBox(sizelistIndex(i1,1),3)];
        stats2.VoxelList{i1,1}=stats1.VoxelList{sizelistIndex(i1,1),1};
        stats2.ConvexHull{i1,1}=stats1.ConvexHull{sizelistIndex(i1,1),1};
    end
    detector_fusion = [];
    detector_split = [];
    detector2_fusion = [];
    detector3_fusion = [];

    stack_after_label(Fullsize_2_mark>0)=0;

    Fullsize_2_mark(Fullsize_2_mark==0)=nan;

    h2=figure; % prepare for 3D figure plot

    % initialize new registration variables
    newc=0;
    l=length(Registration1);
    Registration2=zeros(l,4);
    detector_old=zeros;
    detector_new=zeros;
%     detector_split=zeros;
%     detector3_fusion=zeros;
    detector_numbering=zeros;
    c1=1;
    c2=1;
    c3=1;
    c_numbering=0;
    cc=zeros;
    % tracking for each object
    for i=1:size(stats2.VoxelList,1)
        max_object_intensity1=0;
        max_object_intensity2=0;
        b=stats2.VoxelList{i,1};
        if time+1<10
            add1='00';
        elseif time+1<100
            add1='0';
        end
        % calculate object max/average intensity/size on one/two channels
        threeDimg1=niftiread(strcat('D:\my3D_matlab\3Dimage\Aju2020\18A-09\Aju\','threeDimg_',add1,num2str(time+1),'.nii'));
        threeDimg2=niftiread(strcat('D:\my3D_matlab\3Dimage\Aju2020\18A-09\Myo\','threeDimg_',add1,num2str(time+1),'.nii'));
        threeDimgPixellist1=zeros;
        threeDimgPixellist2=zeros;
        for i1=1:size(b,1)
            threeDimgPixellist1(i1,1)=threeDimg1(b(i1,2),b(i1,1),b(i1,3));
            threeDimgPixellist2(i1,1)=threeDimg2(b(i1,2),b(i1,1),b(i1,3));
            if threeDimg1(b(i1,2),b(i1,1),b(i1,3))>max_object_intensity1
                max_object_intensity1=threeDimg1(b(i1,2),b(i1,1),b(i1,3));
            end
            if threeDimg2(b(i1,2),b(i1,1),b(i1,3))>max_object_intensity2
                max_object_intensity2=threeDimg2(b(i1,2),b(i1,1),b(i1,3));
            end
        end
        threeDimgPixellist1 = sort(threeDimgPixellist1,'descend');
        threeDimgPixellist2 = sort(threeDimgPixellist2,'descend');
        % 80% top-valued pixel to average
%         average_object_intensity1=sum(threeDimgPixellist(threeDimgPixellist>threeDimgPixellist(round(size(threeDimgPixellist,1)*0.8),1)))/(round(size(threeDimgPixellist,1)*0.8)-1);
        average_object_intensity1=sum(threeDimgPixellist1)/size(b,1);
        average_object_intensity2=sum(threeDimgPixellist2)/size(b,1);
        % if a volume of object contains multiple object, take the most
        % representatives
        a=zeros;
        a_t_1=zeros;
        k = boundary(b);
        for i1=1:size(b,1)
            a(i1,1)=Fullsize_2_mark(b(i1,2),b(i1,1),b(i1,3));
        end
        [value,Value_f]=mode(a,'all');
        countnan=sum(isnan(a));
        if countnan>Value_f
            value=nan;
        end

        if time>startpoint % numbering overtracking  %???
            for i1=1:size(b,1)
                a_t_1(i1,1)=Fullsize_1(b(i1,2),b(i1,1),b(i1,3));
            end
            [value_t_1,Value_f_t_1]=mode(a_t_1,'all');
            % find out whether the object has already merged in last time point
            if ~isnan(value_t_1) && isempty(intersect(value_t_1,Registration1(:,1))) && ~isempty(intersect(value_t_1,detector_fusion_old.detector3_fusion)) %merge happed in last time point
                c_numbering=c_numbering+1;
                detector_numbering(c_numbering,1:2)=[value value_t_1];
                value=value_t_1;
                disp(value)
            end
        end
        % find out whether the object has already been tracked in the
        % current time point
        if ~isempty(intersect(value,Registration2(:,1))) % ???
%             disp(value)
            value2=setdiff(a,Registration2(:,1));
            if ~isempty(value2) && size(value2,2)>0 && ~isempty(intersect(value2,Registration1(:,1)))
                value=value2(1,ceil(rand*size(value2,2)));
            end
        end
        % if the representative of an object is NAN, means it is a new
        % obejct, sign new id to it
        if isnan(value)==1

            color=[0 0 0];
            newc=newc+1;
            % document the object in the registration file
            Registration2(l+newc,1)=l+newc;
            Registration2(l+newc,2:4)=stats2.Centroid(i,:);
            value=l+newc;
            % reassign new labels to the sample
            for i1=1:size(b,1)
                Fullsize_2_mark(b(i1,2),b(i1,1),b(i1,3))=value;
                Fullsize_2_2(b(i1,2),b(i1,1),b(i1,3))=value;
            end
            txt=strcat('NEW ',num2str(value));
            detector_new(c1,1)=value;
            c1=c1+1;
            % document the object characteristics
            xlswriter1(l+newc,(time)*2-1)=cellstr('new');
            xlswriter1(l+newc,(time)*2)=cellstr(num2str(l+newc));
            xlswriter3(l+newc,(time)*2)=cellstr(num2str(max_object_intensity1));
            xlswriter4(l+newc,(time)*2)=cellstr(num2str(average_object_intensity1));
            xlswriter5(l+newc,(time)*2)=cellstr(num2str(size(b,1)));
            xlswriter6(l+newc,(time)*2)=cellstr(num2str(stats2.Centroid(i,2)));
            xlswriter7(l+newc,(time)*2)=cellstr(num2str(stats2.Centroid(i,1)));
            xlswriter8(l+newc,(time)*2)=cellstr(num2str(stats2.Centroid(i,3)));

            xlswriter11(l+newc,(time)*2)=cellstr(num2str(max_object_intensity2));
            xlswriter12(l+newc,(time)*2)=cellstr(num2str(average_object_intensity2));
            draw_text(value)=text(b(end,1),b(end,2),b(end,3), txt, 'Rotation',+15);
        % if the representative is not NAN, means we find a tracking
        elseif isnan(value)==0 && value>0

            if isempty(intersect(value,Registration2(:,1)))

                %                 value=index;
                color=map(value,1:3);
%                 Registration2(value,:)=Registration1(value,:);
                Registration2(value,1)=value;
                Registration2(value,2:4)=stats2.Centroid(i,:);
                txt=strcat('OLD ',num2str(value));
                % reassign new labels to the sample
                for i1=1:size(b,1)
                    Fullsize_2_2(b(i1,2),b(i1,1),b(i1,3))=value;
                end
                detector_old(c2,1)=value;
                c2=c2+1;
                xlswriter1(value,(time)*2-1)=cellstr(num2str(value));
                xlswriter1(value,(time)*2)=cellstr(num2str(value));
                xlswriter3(value,(time)*2)=cellstr(num2str(max_object_intensity1));
                xlswriter4(value,(time)*2)=cellstr(num2str(average_object_intensity1));
                xlswriter5(value,(time)*2)=cellstr(num2str(size(b,1)));
                xlswriter6(value,(time)*2)=cellstr(num2str(stats2.Centroid(i,2)));
                xlswriter7(value,(time)*2)=cellstr(num2str(stats2.Centroid(i,1)));
                xlswriter8(value,(time)*2)=cellstr(num2str(stats2.Centroid(i,3)));

                xlswriter11(value,(time)*2)=cellstr(num2str(max_object_intensity2));
                xlswriter12(value,(time)*2)=cellstr(num2str(average_object_intensity2));
                draw_forsure=0;
                for i2=1:size(xlswriter1,2)%????
                    if iscellstr(xlswriter1(value,i2))
                        if string(xlswriter1(value,i2))~="new"
                            if str2num(string(xlswriter1(value,i2)))~=value
                                txt=strcat('OLD',string(xlswriter1(value,i2)),'(',num2str(value),')');
                                draw_text(value)=text(b(end,1),b(end,2),b(end,3), txt, 'Rotation',+15);
                                color=map(str2num(string(xlswriter1(value,i2))),1:3);
                                draw_forsure=1;
                                break
                            end
                            break
                        end
                    end
                end
                if ~draw_forsure
                    draw_text(value)=text(b(end,1),b(end,2),b(end,3), txt, 'Rotation',+15);
                end

                %draw_textt(value)=text(b(end,2),b(end,1),b(end,3), txt, 'Rotation',+15);
            % if the represetative is not NAN, but is zero, means it is a
            % split
            else
                color=map(value,1:3);
                newc=newc+1;
                Registration2(l+newc,1)=l+newc;
                Registration2(l+newc,2:4)=stats2.Centroid(i,:);
                detector_split(c3,1)=value;
                detector_split(c3,2)=l+newc;
                xlswriter10(value,(time)*2)=cellstr(num2str(value));
                xlswriter10(l+newc,(time)*2)=cellstr(num2str(value));
                c3=c3+1;
                % reassign new labels to the sample
                for i1=1:size(b,1)
                    Fullsize_2_2(b(i1,2),b(i1,1),b(i1,3))=l+newc;
                    Fullsize_2_mark(b(i1,2),b(i1,1),b(i1,3))=l+newc;
                end
                xlswriter1(l+newc,1:(time-1)*2)=xlswriter1(value,1:(time-1)*2);
                xlswriter1(l+newc,(time)*2-1)=cellstr(num2str(value));
                xlswriter1(l+newc,(time)*2)=cellstr(num2str(l+newc));
                xlswriter3(l+newc,(time)*2)=cellstr(num2str(max_object_intensity1));
                xlswriter4(l+newc,(time)*2)=cellstr(num2str(average_object_intensity1));
                xlswriter5(l+newc,(time)*2)=cellstr(num2str(size(b,1)));
                xlswriter6(l+newc,(time)*2)=cellstr(num2str(stats2.Centroid(i,2)));
                xlswriter7(l+newc,(time)*2)=cellstr(num2str(stats2.Centroid(i,1)));
                xlswriter8(l+newc,(time)*2)=cellstr(num2str(stats2.Centroid(i,3)));

                xlswriter11(l+newc,(time)*2)=cellstr(num2str(max_object_intensity2));
                xlswriter12(l+newc,(time)*2)=cellstr(num2str(average_object_intensity2));

                for i2=1:size(xlswriter1,2)
                    if iscellstr(xlswriter1(value,i2))
                        if string(xlswriter1(value,i2))~="new"
                            value=str2num(string(xlswriter1(value,i2)));
                            break
                        end
                    end
                end
                txt=strcat('OLD',num2str(value),'=',num2str(l+newc));
                color=map(value,1:3);
                draw_text(l+newc)=text(b(end,1),b(end,2),b(end,3), txt, 'Rotation',+15);
                value=l+newc;
            end

        end
        % plot the object
        trisurf(k,b(:,1),b(:,2),b(:,3),'Facecolor',color,'FaceAlpha',0.3,'Edgecolor',color,'EdgeAlpha',0.3);
        hold on
        if i==1
            draw_text(value)=text(b(end,1),b(end,2),b(end,3), txt, 'Rotation',+15);
        end

    end

    %pointsize = 5;
    %stack_after_label(stack_after_label==0)=nan;
    %scatter3(X(:), Y(:), Z(:), pointsize, stack_after_label(:),'filled');
    %scatter3(X(:), Y(:), Z(:), pointsize, Fullsize_2_mark(:),'filled');
    colormap(map);

    % write the tracking result
    niftiwrite(Registration2,strcat(addr2,'Registration2_tracking','_',t2,'.nii'));
    niftiwrite(Fullsize_2_2,strcat(addr2,'Fullsize_2_aftertracking','_',t2,'.nii'));

    % till here, tracking for old and split object is almost done, below is fusion detection and alarms

    c=1;  %% fusion alarm part1
    for i1=1:size(stats2.VoxelList,1)
        b=stats2.VoxelList{i1,1};
        UNIQUECOUNT=zeros;
        for i2=1:size(b,1)
            UNIQUECOUNT(i2,1)=Fullsize_2_mark(b(i2,2),b(i2,1),b(i2,3));
            if isnan(UNIQUECOUNT(i2,1))
                UNIQUECOUNT(i2,1)=0;
            end
        end
        [C,ia,ic] = unique(UNIQUECOUNT);
        a_counts = accumarray(ic,1);
        value_counts = [C, a_counts];
        [x,y]=size(value_counts);

        if length(C)>1
            detector_fusion(c:c+1,1:size(C,1))=value_counts';
            c=c+2;
        end
    end

    for i1=1:2:size(detector_fusion,1) %% fusion 0 filter
        if detector_fusion(i1,1)==0
            detector_fusion(i1:i1+1,1:end-1)=detector_fusion(i1:i1+1,2:end);
        end
    end

    detector2_fusion=detector_fusion;

    for i1=1:2:size(detector2_fusion,1) %% fusion alarm part2
        for i2=1:size(detector2_fusion,2)
            if intersect(detector2_fusion(i1,i2),Registration2(:,1))
                detector2_fusion(i1:i1+1,i2)=0;
            end
        end
        for i2=1:1:size(detector2_fusion,2)% fusion size filter
            if detector2_fusion(i1+1,i2)<5
                detector2_fusion(i1:i1+1,i2)=0;
            end
        end
    end

    c=1;
    for i1=1:2:length(detector2_fusion(:,1))
        if ~isempty(nonzeros(detector2_fusion(i1,:)))
            detector3_fusion(c:c+1,1:size(detector_fusion,2))=detector_fusion(i1:i1+1,:);
            c=c+2;
        end
    end

    % replace the labels and colors in the figure based on object identify
    for i2=1:2:size(detector3_fusion,1)
        detector3_fusion_exist=0;
        for i1=1:size(detector3_fusion,2)
            if detector3_fusion(i2,i1)>0
                if  isa(draw_text(detector3_fusion(i2,i1)),'matlab.graphics.primitive.Text')
                    draw_text(detector3_fusion(i2,i1)).Color = 'red';
                    if detector3_fusion_exist~=0
                        if detector3_fusion(i2+1,i1)>c
                            detector3_fusion_exist=detector3_fusion(i2,i1);
                            c=detector3_fusion(i2+1,i1);
                        end
                    else
                        detector3_fusion_exist=detector3_fusion(i2,i1);
                        c=detector3_fusion(i2+1,i1);
                    end
                end
            end
        end
        if detector3_fusion_exist==0 %numbering for fusion debug (situation: continue merge that old number is recovered)
            for i1=1:size(detector3_fusion,2)
                if detector3_fusion(i2,i1)>0
                    if ~isempty(intersect(detector3_fusion(i2,i1),detector_numbering))
                        [ii,jj]=find(detector_numbering==detector3_fusion(i2,i1));
                        if size(jj,1)==1
                            value_numbering_recover=detector_numbering(ii,jj+1);

                            txt=draw_text(value_numbering_recover).String;
                            txt=strcat(txt,'M','(',num2str(detector3_fusion(i2,i1)),')');
                            draw_text(value_numbering_recover).String=txt;
                            draw_text(value_numbering_recover).Color='red';
                            detector3_fusion_exist=value_numbering_recover;
                            disp('----')
                            disp(value_numbering_recover)
                        end
                    end
                end
            end
        end
        for i1=1:size(detector3_fusion,2)
            if detector3_fusion(i2,i1)>0 && exist('detector3_fusion_exist','var')
                if  ~isa(draw_text(detector3_fusion(i2,i1)),'matlab.graphics.primitive.Text')
                    xlswriter9(detector3_fusion(i2,i1),time*2-1) = xlswriter1(detector3_fusion_exist,time*2-1);
                end
            end
        end

    end
    % adjust the figure angle and save
    view([0 0 1]);
%     set(gca, 'YDir','reverse')
    xlim([0 I3dw(2)]);
    ylim([0 I3dw(1)]);
    zlim([0 I3dw(3)]);
    hold off
    savefig(h2,strcat(addr2,t2,'_tracking.fig'));
    save(strcat(addr2,'fusion_tracking','_',t2,'.mat'),'detector3_fusion');
    save(strcat(addr2,'split_tracking','_',t2,'.mat'),'detector_split');
    save(strcat(addr2,'detector_numbering','_',t2,'.mat'),'detector_numbering');
    save(strcat(addr2,'draw_text','_',t2,'.mat'),'draw_text');
    close(h2);
    % write the overall object amount to the excel
    xlswriter2(time+1,1)=cellstr(num2str(time));
    xlswriter2(time+1,2)=cellstr(num2str(size(detector_old,1)));
    xlswriter2(time+1,3)=cellstr(num2str(size(detector_new,1)));
    xlswriter2(time+1,4)=cellstr(num2str(size(detector_split,1)));
    xlswriter2(time+1,5)=cellstr(num2str(size(detector3_fusion,1)/2));

    timecount(time)=toc;
    disp(timecount(time))
    clear detector_old detector_new detector_split detector_fusion detector2_fusion detector3_fusion draw_text detector_numbering
end
% save the final tracking results
save(strcat('D:\NEW\',folder,'\','xlswriter1','.mat'),'xlswriter1');
save(strcat('D:\NEW\',folder,'\','xlswriter2','.mat'),'xlswriter2');
save(strcat('D:\NEW\',folder,'\','xlswriter3','.mat'),'xlswriter3');
save(strcat('D:\NEW\',folder,'\','xlswriter4','.mat'),'xlswriter4');
save(strcat('D:\NEW\',folder,'\','xlswriter5','.mat'),'xlswriter5');
save(strcat('D:\NEW\',folder,'\','xlswriter6','.mat'),'xlswriter6');
save(strcat('D:\NEW\',folder,'\','xlswriter7','.mat'),'xlswriter7');
save(strcat('D:\NEW\',folder,'\','xlswriter8','.mat'),'xlswriter8');
save(strcat('D:\NEW\',folder,'\','xlswriter9','.mat'),'xlswriter9');
save(strcat('D:\NEW\',folder,'\','xlswriter10','.mat'),'xlswriter10');
save(strcat('D:\NEW\',folder,'\','xlswriter11','.mat'),'xlswriter11');
save(strcat('D:\NEW\',folder,'\','xlswriter12','.mat'),'xlswriter12');
xlswrite(filename,xlswriter1,1, 'A1');
xlswrite(filename,xlswriter2,2, 'A2');
xlswrite(filename,xlswriter3,3, 'A1');
xlswrite(filename,xlswriter4,4, 'A1');
xlswrite(filename,xlswriter5,5, 'A1');
xlswrite(filename,xlswriter6,6, 'A1');
xlswrite(filename,xlswriter7,7, 'A1');
xlswrite(filename,xlswriter8,8, 'A1');
xlswrite(filename,xlswriter9,9, 'A1');
xlswrite(filename,xlswriter10,10, 'A1');
xlswrite(filename,xlswriter11,11, 'A1');
xlswrite(filename,xlswriter12,12, 'A1');
disp('finished')
    %% tools: plot the ground truth
    clc;
    addr2=strcat('F:\Mo\gt\slices-refined2\');
    Files2=dir(strcat(addr2,'*.tif'));
    I_stack=zeros(280,512,13);
    for i1=1:13
        I=imread(strcat(addr2,Files2(i1).name));
        I_stack(:,:,i1)=I;
    end
    stack_after_BW=logical(I_stack);
    stats = regionprops3(stack_after_BW,'BoundingBox','VoxelList','ConvexHull','Centroid');
    [stack_after_label,orgnum]=bwlabeln(I_stack);
    stack_after_label(stack_after_label==0)=nan;
    h=figure;
%     [X,Y,Z] = ndgrid(1:size(stack_after_label,1), 1:size(stack_after_label,2), 1:size(stack_after_label,3));
%     pointsize = 5;
%     scatter3(X(:), Y(:), Z(:), pointsize, stack_after_label(:),'filled');
%     colormap(map);
    % zlim([0 100]);
    colorbar;
    hold on
    grid on



    for i=1:height(stats)
        Registration(i,:)=stats.Centroid(i,:);
        b=stats.VoxelList{i,1};
        %     for j=length(stats.ConvexHull{i,1})
        %         plot3(a(:,2),a(:,1),a(:,3),'r');
        k = boundary(b);
        hold on
        trisurf(k,b(:,1),b(:,2),b(:,3),'Facecolor','r','FaceAlpha',0.3,'Edgecolor','r','EdgeAlpha',0.3)
        %         hold on
        %     end
%         value=stack_after_label(b(end,1),b(end,2),b(end,3));
%         text(b(end,2),b(end,1),b(end,3), num2str(i), 'Rotation',+15)
    end

    view([0 0 1]);
    set(gca, 'YDir','reverse')
    xlim([0 512]);
    ylim([0 280]);
    zlim([0 13]);
    hold off
%% tools: generate the color map
for i=1:10000
    i1=rand();
    i2=rand();
    i3=rand();
    map(i,1:3)=[i1 i2 i3];
end
save('F:\Mo\my3D_1\Tracking\colormap.mat','map')
