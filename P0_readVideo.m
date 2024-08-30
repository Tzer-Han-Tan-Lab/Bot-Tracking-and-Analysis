vidObj = VideoReader("output_short.mp4");

while hasFrame(vidObj)
    vidFrame = readFrame(vidObj);
    I = rgb2gray( vidFrame );

    imwrite(I, 'test.tif','WriteMode','Append')

    %imshow(I)
    %pause(1/vidObj.FrameRate)
end

%%

Nt=100;

figure()

posArr=cell(1,Nt);
radiusArr=cell(1,Nt);

for t=1:Nt
    im=imread('test-sub.tif',t);

    [centers,radii] = imfindcircles(im,[25 60]);

    imshow(im)
    hold on

    posArr{t}=centers';
    radiusArr{t}=radii;

    for i=1:length(radii)
        viscircles(centers(i,:), radii(i),'Color','r');
    end
    pause(0.1)


end