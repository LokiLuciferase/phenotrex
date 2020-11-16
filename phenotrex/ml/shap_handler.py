from typing import Tuple

import pandas as pd
import numpy as np
import shap
from matplotlib import pyplot as plt

from phenotrex.ml.trex_classifier import TrexClassifier
from phenotrex.util.external_data import Eggnog5TextAnnotator

class ShapHandler:
    """
    This class handles feature arrays and shap values of predictions made with phenotrex,
    and enables plotting of shap values and summaries.

    :param feature_names: All feature names in the model feature space.
    :param used_idxs: Indices into the feature_names array of features
                      actually utilized by the model.
    """
    @classmethod
    def from_clf(cls, clf: TrexClassifier):
        fn = np.array(clf.pipeline.named_steps["vec"].get_feature_names())
        used_fn = [k for k, v in clf.get_feature_weights().items() if v != 0]
        used_idxs = np.where(np.isin(fn, used_fn))[0]
        feature_type = clf.feature_type
        return cls(fn, used_idxs, feature_type=feature_type)

    @staticmethod
    def _fix_shap_force_figure(fig: plt.Figure) -> plt.Figure:
        """
        Replaces the figure annotation in shap force plots with "absent" if value = 0.0 and
        "present" if value = 1.0.

        :param fig: a matplotlib.pyplot.Figure as produced by shap.force_plot.
        :return: The same fig, modified as described.
        """
        ax = fig.gca()
        for c in ax.get_children():
            if isinstance(c, plt.Text):
                t = c.get_text()
                if t.endswith(' = 1.0'):
                    c.set_text(t.replace(' = 1.0', ' present'))
                elif t.endswith(' = 0.0'):
                    c.set_text(t.replace(' = 0.0', ' absent'))
                else:
                    pass
        return fig

    def __init__(self, feature_names: np.ndarray, used_idxs: np.ndarray, feature_type: str = ''):
        self._used_idxs = used_idxs
        self._used_feature_names = feature_names[used_idxs]
        self._feature_type = feature_type
        if feature_type.startswith('eggNOG5'):
            self._text_annotator = Eggnog5TextAnnotator()
            self._feature_taxon = int(feature_type.split('-')[-1])
        else:
            self._text_annotator = None
            self._feature_taxon = None
        self._sample_names = None
        self._used_features = None
        self._used_shaps = None
        self._shap_base_value = None
        self._class_names = ['YES']

    def add_feature_data(
        self,
        sample_names: np.ndarray,
        features: np.ndarray,
        shaps: np.ndarray,
        base_value: float = None
    ):
        """
        Add a new set of feature information to the ShapHandler.

        :param sample_names: an array of sample names; should be globally unique.
        :param features: a feature array of shape (n_sample_names, n_features_in_featurespace).
                         features will be pared down to those used by the model.
        :param shaps: a shap array of shape (n_sample_names, n_features_in_featurespace +1).
                      shaps will be pared down to those produced by the model,
                      mirroring the features.
        :param base_value: If the base value has been split off the shaps before hand, pass it here.
        :return: None
        """
        if base_value is None:
            shaps, base_value = shaps[..., :-1], np.average(shaps[..., -1])

        if self._shap_base_value is None:
            self._shap_base_value = base_value
        else:
            assert np.allclose(self._shap_base_value, base_value), \
                f'Incongruent base values found: {self._shap_base_value} vs. {base_value}.'

        X_used = np.nan_to_num(features[:, self._used_idxs])
        shaps_used = np.nan_to_num(shaps[..., self._used_idxs])

        try:
            X_used = X_used.toarray()
        except AttributeError:
            pass

        if self._sample_names is None:
            self._sample_names = sample_names
        else:
            self._sample_names = np.concatenate([self._sample_names, sample_names])

        if self._used_features is None:
            self._used_features = X_used
        else:
            self._used_features = np.concatenate([self._used_features, X_used])

        if self._used_shaps is None:
            self._used_shaps = shaps_used
        else:
            self._used_shaps = np.concatenate([self._used_shaps, shaps_used])

    def _get_sample_index_with_name(self, sample_name: str) -> int:
        try:
            i = np.where(np.isin(self._sample_names, sample_name))[0][0]
            return i
        except (ValueError, IndexError):
            raise ValueError('Sample label not found among saved explanations.')

    def _get_feature_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Concatenate and return all currently saved features, shaps and sample names.

        :returns: A tuple of saved used features (the actual values),
                  the saved shap values corresponding to the features,
                  and the sample names from which features and shap values were derived.
        """
        try:
            X_agg = self._used_features.astype(float)
            shap_agg = self._used_shaps.astype(float)
        except (ValueError, AttributeError):
            raise RuntimeError('No explanations saved.')
        sample_names = self._sample_names
        return X_agg, shap_agg, sample_names

    def _get_sorted_by_shap_data(
        self, sort_by_idx=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sort features by absolute magnitude of shap values,
        and return sorted features, shap values and feature names.

        :param sort_by_idx: if an index into the sample names is passed,
                            sorting will be based only on this sample's SHAP values.
        :return: Used features, used shaps, and the feature names
                 all sorted by absolute magnitude of shap value.
        """
        X_agg, shap_agg, _ = self._get_feature_data()

        if shap_agg.ndim == 2:
            if sort_by_idx is None:
                feature_sort_inds = np.argsort(np.sum(np.abs(shap_agg), axis=0))[::-1]
            else:
                feature_sort_inds = np.argsort(np.abs(shap_agg[sort_by_idx, :]))[::-1]
        else:
            feature_axis = shap_agg.ndim - 1
            nonfeature_axes = list(range(feature_axis))
            absshap = np.apply_along_axis(np.abs, feature_axis, shap_agg)
            if sort_by_idx is None:
                # sort features by absolute change in shap over all classes and samples
                sort_criterion = np.apply_over_axes(np.sum, absshap, nonfeature_axes)
            else:  # sort features by absolute change in shap over all classes for given sample idx
                sort_criterion = np.sum(absshap[sort_by_idx, ...], axis=0)
            feature_sort_inds = np.squeeze(np.argsort(sort_criterion))[::-1]
        return (X_agg[:, feature_sort_inds], shap_agg[..., feature_sort_inds],
                self._used_feature_names[feature_sort_inds])

    def plot_shap_force(self, sample_name: str, n_max_features: int = 20, **kwargs) -> plt.Figure:
        """
        Create force plot of the sample associated with the given sample name.

        :param sample_name:
        :param n_max_features:
        :param kwargs: additional keyword arguments passed on to `shap.force_plot()`
        :return:
        """
        i = self._get_sample_index_with_name(sample_name)
        X_agg_s, shap_agg_s, feature_names_s = self._get_sorted_by_shap_data(sort_by_idx=i)
        if n_max_features is None:
            n_max_features = len(feature_names_s)

        fig = shap.force_plot(
            base_value=self._shap_base_value,
            shap_values=shap_agg_s[i, :n_max_features],
            features=X_agg_s[i, :n_max_features],
            feature_names=feature_names_s[:n_max_features],
            matplotlib=True,
            show=False,
            text_rotation=45,
            **kwargs
        )
        return self._fix_shap_force_figure(fig)

    def plot_shap_summary(
        self,
        title=None,
        n_max_features: int = 20,
        plot_individual_classes: bool = False,
        **kwargs
    ):
        """
        Create summary plot of shap values over all predicted samples.

        :param title:
        :param n_max_features:
        :param plot_individual_classes:
        :param kwargs: additional keyword arguments passed on to `shap.summary_plot()`
        :return:
        """
        X_agg, shap_agg, _ = self._get_feature_data()

        class_names = self._class_names
        if shap_agg.ndim == 3:
            shap_agg = list(np.swapaxes(shap_agg, 0, 1))
            class_names = class_names[:len(shap_agg)]
            if plot_individual_classes:
                for i, (n, s) in enumerate(zip(class_names, shap_agg)):
                    shap.summary_plot(
                        shap_values=s,
                        features=X_agg,
                        class_names=[f'not {n}', n],
                        feature_names=self._used_feature_names,
                        max_display=n_max_features,
                        show=False,
                        **kwargs
                    )
                    plt.title(f'SHAP Summary for Class {n}')
                    plt.show()

        if title is not None:
            plt.title(title)
        shap.summary_plot(
            shap_values=shap_agg,
            features=X_agg,
            feature_names=self._used_feature_names,
            max_display=n_max_features,
            class_names=class_names,
            title=f'SHAP Summary',
            show=False,
            **kwargs
        )

    def get_shap_force(self, sample_name: str, n_max_features: int = 20) -> pd.DataFrame:
        """
        Create dataframe of the most influential features in sample `sample_name`
        by SHAP value (if multiclass, show SHAP value for each class).

        :param sample_name:
        :param n_max_features:
        :return: a dataframe of the n_max_features most influential features,
                 their value in the sample, and the associated SHAP value(s).
        """
        i = self._get_sample_index_with_name(sample_name)
        X_agg_s, shap_agg_s, feature_names_s = self._get_sorted_by_shap_data(sort_by_idx=i)

        if n_max_features is None:
            n_max_features = len(feature_names_s)

        fns = feature_names_s[:n_max_features]
        feature_vals = X_agg_s[i, :n_max_features]

        if shap_agg_s.ndim == 3:
            shap_agg_s = np.swapaxes(shap_agg_s, 0, 1)
            shap_vals = list(shap_agg_s[:, i, :n_max_features].round(5))
        else:
            shap_vals = [shap_agg_s[i, :n_max_features].round(5), ]
        sample_names = [sample_name] * len(fns)
        df_arrs = [sample_names, fns, feature_vals, *shap_vals]
        df_arrs = [np.array(x) for x in df_arrs]
        df_labels = [
            'Sample',
            'Feature',
            'Feature Presence',
            *[f'SHAP Value (class={x})' for x in self._class_names]
        ][:len(df_arrs)]
        df = pd.DataFrame(df_arrs, index=df_labels).T
        df.index.name = 'rank'
        if self._text_annotator is not None:
            annots = df['Feature'].apply(
                lambda x: self._text_annotator.annotate(self._feature_taxon, x)[1]
            )
            if any(annots):
                df['Feature Annotation'] = annots
        return df.reset_index(drop=False)

    def get_shap_summary(self, n_max_features: int = 50):
        """
        Get summary of features for all predictions,
        sorted by average impact of feature on shap value.

        :param n_max_features:
        :return: a DataFrame of most important SHAP values for samples in the given dataset.
        """
        X_agg_s, shap_agg_s, feature_names_s = self._get_sorted_by_shap_data()
        if n_max_features is None:
            n_max_features = len(feature_names_s)
        columns = ['Feature', 'Mean SHAP If Present', 'Mean SHAP If Absent', 'N(present)', 'N(absent)']
        lines = []
        for i in range(n_max_features):
            feature_name = feature_names_s[i]
            feature = X_agg_s[:, i]
            shaps = shap_agg_s[:, i]
            shaps_where_present = shaps[np.where(feature > 0)[0]]
            shaps_where_absent = shaps[np.where(feature == 0)[0]]
            mean_sv_present = np.average(shaps_where_present)
            mean_sv_absent = np.average(shaps_where_absent)
            n_where_present = len(shaps_where_present)
            n_where_absent = len(shaps_where_absent)
            lines.append([feature_name, mean_sv_present.round(5), mean_sv_absent.round(5),
                          n_where_present, n_where_absent])
        sh_df = pd.DataFrame(lines, columns=columns)
        if self._text_annotator is not None:
            annots = sh_df['Feature'].apply(
                lambda x: self._text_annotator.annotate(self._feature_taxon, x)[1]
            )
            if any(annots):
                sh_df['Feature Annotation'] = annots
        return sh_df
