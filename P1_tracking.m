

load('posArr')

[Rmat,Cmat]=myTracking(posArr(1,1:100),100,0);


figure()
imagesc(Rmat)

%%
figure()
hold on
for i=1:18
    r=Rmat(i,:); c=Cmat(i,:);
    r=r(r~=0); c=c(c~=0);
    plot(r,c)

end


%%

i=3;
figure()
for t=1:377
    im=imread('test.tif',t);

    L=55;
    r1=Rmat(i,t); c1=Cmat(i,t);
    
    box1=im(c1-L:c1+L,r1-L:r1+L);

    imwrite(box1,'bot3.tif','WriteMode','append')

    % imshow(box1)
    % pause(0.1)

end

%%


vidObj = VideoReader("bots3_ss.mp4");
vidArr=cell(1,377);

for t=1:377
    vidFrame = readFrame(vidObj);
    vidArr{t}=vidFrame;
end

%%

vid = VideoWriter('newfile.avi','Motion JPEG AVI');
% vid.Quality=95;
vid.FrameRate = 10;
open(vid);


% figure('Position',[100 100 1200 500])
figure()
for t=1:377

    %vidFrame = readFrame(vidObj,t);

    %im=imread('test.tif',t);
    % for k=1:3
        % subplot(1,3,k)
        imshow(vidArr{t}(:,:,3));
        hold on
        L=50;
        for i=1:3
            r0=Rmat(i,t); c0=Cmat(i,t);
            plot(r0+[-1 1 1 -1 -1]*L,c0+[1 1 -1 -1 1]*L,'r')
        end

    frame = getframe(gcf);
    writeVideo(vid,frame);
    clf
%     end
end

close(vid)