function [Rmat,Cmat]=myTracking(posArr,max_linking_distance,max_gap_closing)
% posArr is a cell of Nt rc-vector, each rc-vector is 2-by-len, where len is
% the number of ps detected on that particular frame
% max_linking_distance and max_gap_closing are parameters for simpletracker

Nt=length(posArr);

points{Nt}=[];

for t=1:Nt
    points{t}=posArr{t}';
end

%max_linking_distance = 15; % some value between 5 to 10 seems reasonable
%max_gap_closing = 0;
debug = true;

[tracks, adjacency_tracks] = simpletracker(points,...
    'MaxLinkingDistance', max_linking_distance, ...
    'MaxGapClosing', max_gap_closing, ...
    'Debug', debug);

n_tracks = numel(tracks);
colors = hsv(n_tracks);

all_points = vertcat(points{:});

% figure(10) %Hugh edit**
hold on
for i_track = 1 : n_tracks
    % We use the adjacency tracks to retrieve the points coordinates. It
    % saves us a loop.
    track = adjacency_tracks{i_track};
    track_points = all_points(track, :);
%     plot(track_points(:,1), track_points(:, 2), 'Color', colors(i_track, :))
end

[Rmat,Cmat]=getRCmat(points,tracks,adjacency_tracks);

function [Rmat,Cmat]=getRCmat(points,tracks,adjacency_tracks)

% coordinates are in row-col

N=length(tracks);
T=length(tracks{1});

% record row and col of the N tracks identified, across T frames
Rmat=zeros(N,T);
Cmat=zeros(N,T);

for i=1:N
    
    tracki=tracks{i};
    fidx=find(~isnan(tracki)); 
    % tidx tells you the time in which the position occur
    
    full_fidx=fidx(1):fidx(end);
    rc_idx=tracki(full_fidx);
    
    len=length(full_fidx);
    rc_pos=zeros(len,2);
    for k=1:len
        rc_i=rc_idx(k);
        if ~isnan(rc_i)
            rc_pos(k,:)=points{full_fidx(k)}(rc_i,:);
        end
    end
    
    Rmat(i,full_fidx)=rc_pos(:,1)';
    Cmat(i,full_fidx)=rc_pos(:,2)';
    
end