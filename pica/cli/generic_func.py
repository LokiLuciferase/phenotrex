from pica.io.flat import (load_training_files, write_weights_file, write_misclassifications_file,
                          write_cccv_accuracy_file)
from pica.io.serialization import save_classifier
from pica.util.logging import get_logger
from pica.ml.classifiers import TrexSVM, TrexXGB

CLF_MAPPER = {'svm': TrexSVM, 'xgb': TrexXGB}
logger = get_logger("trex", verb=True)


def generic_train(type, genotype, phenotype, verb, weights, out, n_features=None, *args, **kwargs):
    """Generic function for model training."""
    training_records, *_ = load_training_files(genotype_file=genotype,
                                               phenotype_file=phenotype,
                                               verb=verb)
    clf = CLF_MAPPER[type](*args, **kwargs)
    reduce_features = True if n_features is not None else False
    clf.train(records=training_records, reduce_features=reduce_features, n_features=n_features)
    if weights:
        weights = clf.get_feature_weights()
        weights_file_name = f"{out}.rank"
        write_weights_file(weights_file=weights_file_name, weights=weights)
    save_classifier(obj=clf, filename=out, verb=verb)


def generic_cv(type, genotype, phenotype, folds, replicates, threads,
               verb, groups=None, rank=None, out=None, n_features=None,  *args, **kwargs):
    """Generic function for model CV."""
    training_records, *_ = load_training_files(genotype_file=genotype,
                                               phenotype_file=phenotype,
                                               groups_file=groups,
                                               selected_rank=rank,
                                               verb=verb)
    clf = CLF_MAPPER[type](*args, **kwargs)
    reduce_features = True if n_features is not None else False
    use_groups = groups is not None
    score_mean, score_sd, misclass = clf.crossvalidate(records=training_records, cv=folds,
                                                       n_replicates=replicates, groups=use_groups,
                                                       n_jobs=threads,
                                                       reduce_features=reduce_features,
                                                       n_features=n_features)
    logger.info(f"CV score: {score_mean} +/- {score_sd}")
    if out is not None:
        write_misclassifications_file(out, training_records, misclass, use_groups=use_groups)


def generic_cccv(type, genotype, phenotype, folds, replicates, threads, comple_steps, conta_steps,
                 verb, groups=None, rank=None, out=None, n_features=None,  *args, **kwargs):
    """Generic function for model CCCV."""
    assert groups is None, 'Usage of LOGO in CCCV not currently implemented.'
    training_records, *_ = load_training_files(genotype_file=genotype,
                                               phenotype_file=phenotype,
                                               verb=verb)
    clf = CLF_MAPPER[type](*args, **kwargs)
    reduce_features = True if n_features is not None else False
    cccv = clf.crossvalidate_cc(records=training_records, cv=folds, n_replicates=replicates,
                                comple_steps=comple_steps, conta_steps=conta_steps,
                                n_jobs=threads, reduce_features=reduce_features, n_features=n_features)
    write_cccv_accuracy_file(out, cccv)
