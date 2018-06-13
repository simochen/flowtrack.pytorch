dataType = '';
addpath('coco/MatlabAPI');

mkdir('mat')

annTypes = { 'instances', 'captions', 'person_keypoints' };
annType=annTypes{3}; % specify dataType/annType

for mode = 0:1
    
    if mode == 0
        dataType= 'val2017';
        annFile=sprintf('annotations/%s_%s.json',annType,dataType);
    else
        dataType = 'train2017';
        annFile=sprintf('annotations/%s_%s.json',annType,dataType);
    end
    
    coco=CocoApi(annFile);
    %%
    my_anno = coco.data.annotations;
    %%
    prev_id = -1;
    p_cnt = 1;
    cnt = 0;
    coco_kpt = [];
    
    for i = 1:1:size(my_anno,2)
        
        curr_id = my_anno(i).image_id;
        if(curr_id == prev_id)
            p_cnt = p_cnt + 1;
        else
            p_cnt = 1;
            cnt = cnt + 1;
            coco_kpt(cnt).image_id = curr_id;
            coco_kpt(cnt).img_width = coco.loadImgs(curr_id).width;
            coco_kpt(cnt).img_height = coco.loadImgs(curr_id).height;
        end
        coco_kpt(cnt).annorect(p_cnt).bbox = my_anno(i).bbox;
        coco_kpt(cnt).annorect(p_cnt).segmentation = my_anno(i).segmentation;
        coco_kpt(cnt).annorect(p_cnt).area = my_anno(i).area;
        coco_kpt(cnt).annorect(p_cnt).id = my_anno(i).id;
        coco_kpt(cnt).annorect(p_cnt).iscrowd = my_anno(i).iscrowd;
        coco_kpt(cnt).annorect(p_cnt).keypoints = my_anno(i).keypoints;
        coco_kpt(cnt).annorect(p_cnt).num_keypoints = my_anno(i).num_keypoints;
         
        prev_id = curr_id;
        
        fprintf('%d/%d \n', i, size(my_anno, 2));
    end
    %%
    if mode == 0
        coco_val = coco_kpt;
        save('mat/coco_val.mat', 'coco_val');
    else
        save('mat/coco_kpt.mat', 'coco_kpt');
    end
    
end