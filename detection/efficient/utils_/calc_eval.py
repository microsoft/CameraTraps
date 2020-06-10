"""
COCO-Style Evaluations

put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""
import io
import sys
import torch

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import invert_affine, postprocess

def evaluate_mAP(imgs, imgs_ids, framed_metas, regressions, \
                 classifications, anchors, threshold=0.05, nms_threshold=0.5):
    '''
    Inputs: Images, Image IDs, Framed Metas (Resizing stats), prredictions
    Output: results
    '''
    results = [] # This is used for storing evaluation results.
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    preds = postprocess(imgs,
                        torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regressions.detach(), classifications.detach(),
                        regressBoxes, clipBoxes,
                        threshold, nms_threshold)

    if not preds:
        return

    preds = invert_affine(framed_metas, preds)
    for i, _ in enumerate(preds):
        scores = preds[i]['scores']
        class_ids = preds[i]['class_ids']
        rois = preds[i]['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                if score < threshold:
                    break

                image_result = {
                    'image_id': imgs_ids[i],
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)
    return results

def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')

    # pattern = r"(.+)@\[ (.+) \| (.+) \| (.+) \] = (.+)"
    category_results = dict()

    for catgry in coco_gt.loadCats(coco_gt.getCatIds()):
        print("mAP metrics for Category --> ", catgry['name'])
        category_results[catgry['name']] = dict()
        coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.params.catIds = [catgry['id']]
        coco_eval.evaluate()
        coco_eval.accumulate()
        _stdout = sys.stdout # Start getting console output
        sys.stdout = io.StringIO()
        coco_eval.summarize()
        coco_summary = sys.stdout.getvalue()
        sys.stdout = _stdout # Stop getting console output
        coco_summary = str(coco_summary).strip()
        lines = coco_summary.split('\n')
        for lin in lines:
            chunks = lin.split('=')
            metric = " = ".join(chunks[:-1]).strip()
            score = float(chunks[-1].strip())
            category_results[catgry['name']][metric] = score
    return category_results

def calc_mAP_fin(project_name='shape',
                 set_name='val',
                 evaluation_pred_file='datasets/shape/predictions/instances_bbox_results.json',
                 val_gt = 'datasets/shape/annotations/instances_val.json',
                 max_images = 100000):
    coco_gt = COCO(val_gt)
    image_ids = coco_gt.getImgIds()[:max_images]
    return _eval(coco_gt, image_ids, evaluation_pred_file)
