# Read-only ICFG support and holdout feasibility

Inspected while V016 CUHK runs. No ICFG model,training run,derived annotation
or finalized validation split has been created. No image content,weights or
files were hashed. No retrieval test scores were read.

The existing conflict is confirmed in official train_reid.json: one path
has PID1 and PID5. Quarantining BOTH records would leave34672unique images
and3102identities,with exactly one caption and one backtranslation per image.
Raw annotations/images remain unchanged. The filename's test/ directory is
not a split indicator; these records belong to the official train split.

Prospective fixed holdout rule used for this structural calculation only:
sorted training PIDs -> numpy RandomState(1).permutation -> first310IDs for
validation. After quarantine,that gives:

| Pool | Images/captions | IDs |
|---|---:|---:|
| Remaining train |31289|2792|
| Train-ID-disjoint prospective val |3383|310|

29221/31289prospective train images (93.39065%) have at least one distinct
same-ID image with different peak F/S/B view and circular angle gap>=60.
Current CUHK pool has20702/34054eligible images (60.79168%). These are
structural support counts,not accuracy predictions.

ICFG full quarantined train pool images-per-PID distribution:
2:1,3:2,4:4,5:19,6:228,7:253,8:322,9:262,10:282,11:305,12:302,
13:289,14:235,15:200,16:180,17:170,18:48.

If this candidate dataset is chosen later,first freeze the split and
quarantine policy in an independent preparation directory,then measure its
own five-epoch E0. Both E0 and method must exclude the holdout identities.
Derived val images retain source_split=train when checking OEFormer metadata;
do not falsely require them to appear in OEFormer's official val field.
Official test remains untouched and cannot guide either split or tuning.
