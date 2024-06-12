# Code for Paper "Reevaluation of Inductive Link Prediction"

## Installation

```bash
conda create -n indeval python=3.8
conda activate indeval
pip install torch pykeen=1.10.2
```

## Baseline

Run `python baseline.py`

## Evaluate your approach

Run `python eval.py --dataset {WN18RR,fb237,nell,ilpc_large,ilpc_small} --version {v1,v2,v3,v4} --path PATH` which outputs hits@10 and MRR for sampling-based, non-sampling and tmn evaluation protocol.

where PATH either points to a text file (ending .txt) containing an prediction file analogous to AnyBURL

```text
{test_h1} {test_r1} {test_t1}
Heads: {pred_h1}\t{score_pred_h1}\t{pred_h2}\t{score_pred_h2}\t ...
Tails: {pred_t1}\t{score_pred_t1}\t{pred_t2}\t{score_pred_t2}\t ...
{test_h2} {test_r2} {test_t2}
Heads: {pred_h1}\t{score_pred_h1}\t{pred_h2}\t{score_pred_h2}\t ...
Tails: {pred_t1}\t{score_pred_t1}\t{pred_t2}\t{score_pred_t2}\t ...
...
```

for example

```text
/m/01zc2w /education/field_of_study/students_majoring./education/education/major_field_of_study /m/0jjw
Heads: /m/01zc2w	0.75	/m/02j62	0.2222222222222222	/m/0fdys	0.16666666666666666	/m/01mkq	0.12320040690821256	/m/04_tv	0.12320040626773657	/m/05qjt	0.12320040626768969	/m/036nz	0.12320040579710145	/m/02vxn	0.12320040579710145	/m/040p_q	0.12318840579710146	/m/023907r	0.12318840579710146	/m/0_jm	0.12280701754385964	/m/02bjrlw	0.12000470588235293	/m/06mq7	0.12000470588235293	/m/0hcr	0.12	/m/03ytc	0.12	/m/026bk	0.10119047619047619	
Tails: /m/0jjw	0.75	/m/02j62	0.2222222222222222	/m/01mkq	0.21428571428571427	/m/0fdys	0.16668095238095237	/m/05qjt	0.16667898673531653	/m/036nz	0.16667898550724636	/m/02vxn	0.16667898550724636	/m/040p_q	0.16666666666666666	/m/023907r	0.16666666666666666	/m/0_jm	0.123200686604119	/m/04_tv	0.12320068649885584	/m/02bjrlw	0.12319465579710145	/m/06mq7	0.12319465579710145	/m/026bk	0.12318840579710146	/m/0hcr	0.12318840579710146	/m/03ytc	0.12318840579710146	/m/07c52	0.12280701754385964	/m/05qdh	0.12280701754385964	
```