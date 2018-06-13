addpath('coco/MatlabAPI/');
% addpath('../testing/util');

mkdir('images/mask2017')
vis = 0;

for mode = 0:1
    
    if mode == 1 
        load('mat/coco_kpt.mat');
    else
        load('mat/coco_val.mat');
        coco_kpt = coco_val;
    end
    
    L = length(coco_kpt);
    %%
    
    for i = 1:L
        if mode == 1
            img_paths = sprintf('images/train2017/%012d.jpg', coco_kpt(i).image_id);
            img_name1 = sprintf('images/mask2017/mask_all_%012d.png', coco_kpt(i).image_id);
            img_name2 = sprintf('images/mask2017/mask_miss_%012d.png', coco_kpt(i).image_id);
        else
            img_paths = sprintf('images/val2017/%012d.jpg', coco_kpt(i).image_id);
            img_name1 = sprintf('images/mask2017/mask_all_%012d.png', coco_kpt(i).image_id);
            img_name2 = sprintf('images/mask2017/mask_miss_%012d.png', coco_kpt(i).image_id);
        end
        
        if 1
%         try
%             imread(img_name1);
%             imread(img_name2);
%             continue;
%         catch
            display([num2str(i) '/ ' num2str(L)]);
            %joint_all(count).img_paths = RELEASE(i).image_id;
            [h,w,~] = size(imread(img_paths));
            mask_all = false(h,w);
            mask_miss = false(h,w);
            flag = 0;
            for p = 1:length(coco_kpt(i).annorect)
                seg = coco_kpt(i).annorect(p).segmentation;
                %if this person is annotated
                if iscell(seg)
                    [X,Y] = meshgrid( 1:w, 1:h );
                    for k = 1:length(seg)
                        mask = inpolygon( X, Y, seg{k}(1:2:end), seg{k}(2:2:end));
                        mask_all = or(mask, mask_all);

                        if coco_kpt(i).annorect(p).num_keypoints <= 0
                            mask_miss = or(mask, mask_miss);
                        end
                    end
                else % is struct
                    %display([num2str(i) ' ' num2str(p)]);
                    mask_crowd = logical(MaskApi.decode( seg ));
                    temp = and(mask_all, mask_crowd);
                    mask_crowd = mask_crowd - temp;
                    flag = flag + 1;
                    coco_kpt(i).mask_crowd = mask_crowd;
                    mask_all = or(mask_all, mask_crowd);
                    mask_miss = or(mask_miss, mask_crowd);
                end
            end
            mask_miss = not(mask_miss);
            
            coco_kpt(i).mask_all = mask_all;    % all people (annotated and not annotated)
            coco_kpt(i).mask_miss = mask_miss;  % all except person not annotated
            
            if mode == 1
                img_name = sprintf('images/mask2017/mask_all_%012d.png', coco_kpt(i).image_id);
                imwrite(mask_all,img_name);
                img_name = sprintf('images/mask2017/mask_miss_%012d.png', coco_kpt(i).image_id);
                imwrite(mask_miss,img_name);
            else
                img_name = sprintf('images/mask2017/mask_all_%012d.png', coco_kpt(i).image_id);
                imwrite(mask_all,img_name);
                img_name = sprintf('images/mask2017/mask_miss_%012d.png', coco_kpt(i).image_id);
                imwrite(mask_miss,img_name);
            end
            
%             if flag == 1 && vis == 1
%                 im = imread(img_paths);
%                 mapIm = mat2im(mask_all, jet(100), [0 1]);
%                 mapIm = mapIm*0.5 + (single(im)/255)*0.5;
%                 figure(1),imshow(mapIm);
%                 mapIm = mat2im(mask_miss, jet(100), [0 1]);
%                 mapIm = mapIm*0.5 + (single(im)/255)*0.5;
%                 figure(2),imshow(mapIm);
%                 mapIm = mat2im(mask_crowd, jet(100), [0 1]);
%                 mapIm = mapIm*0.5 + (single(im)/255)*0.5;
%                 figure(3),imshow(mapIm);
%                 pause;
%                 close all;
%             elseif flag > 1
%                 display([num2str(i) ' ' num2str(p)]);
%             end
        end
    end
    
    if mode == 1 
        save('mat/coco_kpt_mask.mat', 'coco_kpt', '-v7.3');
    else
        coco_val = coco_kpt;
        save('mat/coco_val_mask.mat', 'coco_val', '-v7.3');
    end
    
end