import pytest

from vcmr.evaluation import evaluate_predictions, temporal_iou, paired_bootstrap, save_predictions_jsonl, load_predictions_jsonl
from vcmr.data import episode_id, show_id


def gt(qid=1, video='v1', span=(10,20), caption='a person stands', kind='v'):
    return {'desc_id': qid, 'vid_name': video, 'ts': list(span), 'desc': caption, 'type': kind}


def test_temporal_iou_overlapping_disjoint_and_invalid():
    assert temporal_iou((10,20),(10,20)) == 1
    assert temporal_iou((10,20),(15,25)) == pytest.approx(1/3)
    assert temporal_iou((10,20),(20,25)) == 0
    assert temporal_iou((10,10),(5,15)) == 0
    assert temporal_iou((10,float('nan')),(5,15)) == 0


def test_wrong_video_never_matches_even_with_identical_span():
    result = evaluate_predictions({'1': [['wrong',10,20,1.]]}, [gt()])
    assert result['full']['R1@0.5'] == 0


def test_multiple_positive_intervals_and_video_identities():
    rows = [gt(), gt(span=(30,40)), gt(video='v2',span=(50,60))]
    result = evaluate_predictions({'1': [['v2',50,60,1.]]}, rows)
    assert result['full']['n_queries'] == 1
    assert result['full']['R1@0.7'] == 100
    assert result['diagnostics']['n_input_gt_records'] == 3


def test_predictions_are_ranked_globally_by_score():
    predictions = {'1': [['v1',10,20,.1], ['v2',10,20,.9]]}
    result = evaluate_predictions(predictions, [gt()])
    assert result['full']['R1@0.5'] == 0
    assert result['full']['R5@0.7'] == 100


def test_caption_duplicates_are_distinct_queries_but_duplicate_ids_collapse():
    rows = [gt(),gt(),gt(qid=2)]
    result = evaluate_predictions({'1':[['v1',10,20,1.]]},rows)
    assert result['full']['n_queries'] == 2
    assert result['full']['R1@0.5'] == 50
    assert result['diagnostics']['n_missing_prediction_queries'] == 1
    with pytest.raises(ValueError, match='Conflicting captions'):
        evaluate_predictions({}, [gt(),gt(caption='different query')])


def test_duplicate_predictions_consume_rank_slots_and_visual_subset():
    rows = [gt(),gt(qid=2,kind='t')]
    preds = {'1': [['wrong',0,1,3],['wrong',0,1,2],['v1',10,20,1]],'2':[['v1',10,20,1]]}
    result = evaluate_predictions(preds,rows,ranks=(1,2,3))
    assert result['full']['R1@0.5'] == 50
    assert result['visual_only']['R2@0.5'] == 0
    assert result['visual_only']['R3@0.5'] == 100


def test_empty_predictions_and_invalid_prediction_validation():
    assert evaluate_predictions({},[gt()])['full']['R100@0.5'] == 0
    for item in [['v1',20,10,1],['v1',-1,10,1],['v1',10,20,float('inf')]]:
        with pytest.raises(ValueError):
            evaluate_predictions({'1':[item]},[gt()])


def test_bootstrap_is_paired_clustered_and_directional():
    rows = [gt(),gt(qid=2),gt(qid=3,video='v3')]
    a = {str(r['desc_id']): [['wrong',10,20,1]] for r in rows}
    b = {str(r['desc_id']): [[r['vid_name'],10,20,1]] for r in rows}
    result = paired_bootstrap(a,b,rows,iterations=50)
    assert result['n_video_clusters'] == 2
    assert result['delta_b_minus_a_pp'] == 100
    assert result['ci95_pp'] == [100,100]


def test_prediction_schema_roundtrip(tmp_path):
    path=tmp_path/'predictions.jsonl'
    save_predictions_jsonl(path,{1:[['v1',10,20,.5]]})
    assert load_predictions_jsonl(path)=={'1':[['v1',10.,20.,.5]]}


def test_episode_grouping_preserves_all_segments_and_shows():
    assert episode_id('friends_s09e23-24_seg02_clip_33') == 'friends_s09e23-24'
    assert show_id('friends_s09e23-24_seg02_clip_33') == 'friends'
    assert show_id('s01e10_seg01_clip_02') == 'unnamed_show'
