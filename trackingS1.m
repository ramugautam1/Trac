%% tools: make the 3Dimage for tracking

%-------------------------------------------------------------------------------------------------------------------%
%% step1: process the segmentation results to tracking candidates
clear
load('/home/nirvan/Desktop/Projects/MATLAB CODES/colormap.mat','map')
% t1=[2, 29, 30];
t1=1;
t2=42;
%source_data=niftiread(strcat('F:\Mo\gt\Myo-cherry and Aju-GFP 2020.12.21Sample\source\','sqh-cherry_jub-gfp 18A-02.nii'));
I3dw=[512, 280, 15];%size(source_data);
I3d=[32,35,I3dw(3)];
for time=t1:1
    disp(time)
    tt=num2str(time);
    addr=strcat('/home/nirvan/Desktop/Projects/EcadMyo_08_all/Segmentation_Result_EcadMyo_08/EcadMyo_08/FC-DenseNet/',tt,'/');
    addr2=strcat('/home/nirvan/Desktop/Projects/EcadMyo_08_all/EcadMyo_08_tr/',tt,'/');
    if ~exist(addr2,'dir')
        mkdir(addr2);
    end
    Files1=dir(strcat(addr,'*.nii'));
    Fullsize=zeros(512,280,15);
    Fullsize_regression=zeros(512,280,15);
    Fullsize_input=zeros(512,280,15);
    Weights=zeros(512,280,15,64);
    c_file=1;

    for i1=1:I3d(1):I3dw(1)-I3d(1)+1
            for i2=1:I3d(2):I3dw(2)-I3d(2)+1
                V = niftiread(strcat(addr,Files1(c_file).name));
                V=1-V;   %Because the class dictionary was defined that way (background =1 and foreground =0)
                V2= uint8(V*255);
                V3= imresize3(V2, I3d,'linear');

                a=i1;
                b=i1+I3d(1)-1;
                c=i2;
                d=i2+I3d(2)-1;
                Fullsize(a:b, c:d, :)=V3;

                V = niftiread(strcat(addr,Files1(c_file+1).name));
                for iy=1:64  %%all=256
                    V2= double(V(:,:,:,iy));
                    V3= imresize3(V2, I3d,'linear');
                    Weights(a:b,c:d,:,iy)=V2;
                end

                V = niftiread(strcat(addr,Files1(c_file+2).name));
                V3= imresize3(V, I3d,'linear');
                Fullsize_input(a:b, c:d, :)=V3;
                c_file=c_file+4;
            end
    end


    %Remove the small itty-bitty masks
% % %        niftiwrite(Fullsize,strcat(addr2,'testm','_',tt,'.nii'));
    Fullsize2=logical(Fullsize);
 
    for it=1:size(Fullsize,3)
        img=Fullsize2(:,:,it);
        [f,orgnum] = bwlabel(img);
        g = regionprops(f, 'Area');
        area_values=[g.Area];
%         idx = find ((25<= area_values & 800>= area_values));
        idx = find ((5<= area_values));
        h = ismember (f, idx);
        Fullsize2(:,:,it)=h;
    end
    Fullsize2=double(Fullsize2);


    stack_after=Fullsize2;
    [y, x, z] = size(Fullsize);

    stack_after_BW=logical(stack_after);

% %     [stack_after_label,orgnum]=bwlabeln(Fullsize2);
% %     CC = bwconncomp(Fullsize2,6);
% %     stats = regionprops3(CC,'BoundingBox','VoxelList','ConvexHull','Centroid','Volume');
% %     j=height(stats);

    [stack_after_label,orgnum]=bwlabeln(stack_after, 6);
    CC = bwconncomp(stack_after,6);
    stats1 = regionprops3(CC,'BoundingBox','VoxelList','ConvexHull','Centroid');
%     %stack_after(stack_after==0)=nan;

% % %     niftiwrite(stack_after_label,strcat(addr2,'Fullsize_labelx','_',tt,'.nii'));

    disp(stats1(:,:))


    for i=1:height(stats1)
        b=stats1.VoxelList{i,1};
        [x,y]=size(b);
        for i1=1:x
            stack_after_label(stack_after_label(b(i1,2),b(i1,1),b(i1,3))>0)=i;
        end
    end
%----------------------
    stack_after_label(stack_after_label==0)=nan;
    h=figure;
    [X,Y,Z] = ndgrid(1:size(stack_after_label,1), 1:size(stack_after_label,2), 1:size(stack_after_label,3));
    pointsize = 5;
%     scatter3(X(:), Y(:), Z(:), pointsize, stack_after_label(:),'filled');
    colormap(map);
    % zlim([0 100]);
    colorbar;
    hold on
    grid on

    for i=1:height(stats1)

        b=stats1.VoxelList{i,1};

        k = boundary(b);
        hold on
        value=i;
        Registration(value,1)=value;
        Registration(value,2:4)=stats1.Centroid(i,:);

            c1=[255/255 0 0];
            c2=[150/255 100/255 100/255];

        trisurf(k,b(:,1),b(:,2),b(:,3),'Facecolor',c1,'FaceAlpha',0.5,'Edgecolor',c1,'EdgeAlpha',0.5)

        text(b(end,1),b(end,2),b(end,3), num2str(value), 'Rotation',+15, 'Color', c2)
    end


    hold off
    view([1 1 1]);
%     set(gca, 'YDir','reverse')
    xlim([0 280]);
    ylim([0 512]);
    zlim([0 15]);
%------------------

% % %     niftiwrite(Fullsize2,strcat(addr2,'Fullsizex','_',tt,'.nii'));

    
%     niftiwrite(stack_after_label,strcat(addr2,'Fullsize_label','_',tt,'.nii'));

% % %     niftiwrite(Registration,strcat(addr2,'Registration','_',tt,'.nii'));

%    niftiwrite(Fullsize_regression,strcat(addr2,'Fullsize_regression','_',tt,'.nii'));
% % %     niftiwrite(Weights,strcat(addr2,'Weights','_',tt,'.nii'));

% % %     savefig(h,strcat(addr2,tt,'_3Dconnection2.fig'));
% % %     saveas(h,strcat(addr2,tt,'_3Dconnection2.png'))
    close(h);

end
disp('finish')