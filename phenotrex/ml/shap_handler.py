from pathlib import Path
from typing import Tuple, Union, List

import numpy as np
import shap
from matplotlib import pyplot as plt

from phenotrex.ml.trex_classifier import TrexClassifier


class ShapHandler:
    """
    This class handles feature arrays and shap values of predictions made with phenotrex,
    and enables plotting of shap values and summaries.
    """
    @classmethod
    def from_clf(cls, clf: TrexClassifier):
        fn, fn_idx = zip(*clf.pipeline.named_steps["vec"].get_feature_names())
        fn = np.array(fn)
        used_fn = [k for k, v in clf.get_feature_weights().items() if v != 0]
        used_idxs = np.where(np.isin(fn, used_fn))[0]
        return cls(fn, used_idxs)

    def __init__(self, fn: np.ndarray, used_idxs: np.ndarray):
        self._used_idxs = used_idxs
        self._used_feature_names = fn[used_idxs]

        self._sample_names = None
        self._used_features = None
        self._used_shaps = None
        self._shap_base_value = None

    def add_data(self, sample_names: np.ndarray, features: np.ndarray, shaps: np.ndarray,
                 base_value: float):
        """
        Add a new set of feature information + shap values to the ShapHandler.

        :param sample_names: an array of sample names; should globally unique.
        :param features: a feature array of shape (n_sample_names, n_features_in_featurespace).
                         features will be pared down to those used by the model.
        :param shaps: a shap array of shape (n_sample_names, n_features_in_featurespace +1).
                          shaps will be pared down to those produced by the model, mirroring the features.
        :param base_value:
        :return: None
        """
        assert features.ndim == 2 and shaps.ndim == 2, 'Non-2D arrays supplied.'
        if self._shap_base_value is None:
            self._shap_base_value = base_value
        else:
            assert np.allclose(self._shap_base_value, base_value), \
                f'Incongruent base values found: {self._shap_base_value} vs. {base_value}.'

        X_used = np.nan_to_num(features[:, self._used_idxs])
        shaps_used = np.nan_to_num(shaps[:, self._used_idxs])

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
            i = np.where(np.isin(self._sample_names, np.array([sample_name])))[0][0]
            return i
        except (ValueError, IndexError):
            raise ValueError('Sample label not found among saved explanations.')

    def _get_feature_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Concatenates and returns all currently saved features, shaps and sample names.

        """
        try:
            X_agg = self._used_features.astype(float)
            shap_agg = self._used_shaps.astype(float)
        except ValueError:
            raise ValueError('No explanations saved.')
        sample_names = self._sample_names
        return X_agg, shap_agg, sample_names

    def _get_sorted_by_shap_data(self, sort_by_idx=None):
        """
        Sort features by absolute magnitude of shap values,
        and return sorted features, shap values and feature names.

        """
        X_agg, shap_agg, _ = self._get_feature_data()
        if sort_by_idx is None:
            sort_inds = np.argsort(np.sum(np.abs(shap_agg), axis=0))[::-1]
        else:
            sort_inds = np.argsort(np.abs(shap_agg[sort_by_idx, :]))[::-1]
        return X_agg[:, sort_inds], shap_agg[:, sort_inds], self._used_feature_names[sort_inds]

    def plot_shap_force(self, sample_name: str, out_file_path: Union[str, Path] = None, **kwargs):
        """
        Create force plot of the sample associated with the given sample name.

        :param sample_name:
        :param out_file_path:
        :param kwargs: additional keyword arguments passed on to `shap.force_plot()`
        :return:
        """
        counts, shaps, sample_names = self._get_feature_data()
        i = self._get_sample_index_with_name(sample_name)

        sf, ss = counts[i, :], shaps[i, :]
        shap.force_plot(base_value=self._shap_base_value, shap_values=ss, features=sf,
                        feature_names=self._used_feature_names, matplotlib=True, **kwargs)

        if out_file_path is not None:
            out_file_path = Path(str(out_file_path))
            assert not out_file_path.is_file(), 'The output file already exists.'
            plt.savefig(out_file_path)
        else:
            plt.show()

    def plot_shap_summary(self, out_file_path: str = None, title=None, n_max_features: int = 20,
                          **kwargs):
        """
        Create summary plot of shap values over all predicted samples.

        :param out_file_path:
        :param title:
        :param n_max_features:
        :param kwargs: additional keyword arguments passed on to `shap.summary_plot()`
        :return:
        """
        X_agg, shap_agg, _ = self._get_feature_data()

        if title == 'auto':
            title = f'Feature Summary of Model {self._clf},\n' \
                    f'computed on {len(self._sample_names)} samples'
        if title is not None:
            plt.title(title)
        shap.summary_plot(shap_values=shap_agg,
                          features=X_agg,
                          feature_names=self._used_feature_names,
                          max_display=n_max_features,
                          **kwargs)
        if out_file_path is not None:
            out_file_path = Path(str(out_file_path))
            assert not out_file_path.is_file(), 'The output file already exists.'
            plt.savefig(out_file_path)
        else:
            plt.show()

    def get_shap_force(self, sample_name: str, n_max_features: int = 20) -> Tuple[List[Tuple], float]:
        """
        Create List of tuples containing features, their value in the given sample
        and the resulting change in shap value. Also returns the total shap value of the given sample.
        The list is sorted by the influence of the feature on the shap values of the given sample.

        :param sample_name:
        :param n_max_features:
        :return:
        """
        i = self._get_sample_index_with_name(sample_name)
        X_agg_s, shap_agg_s, feature_names_s = self._get_sorted_by_shap_data(sort_by_idx=i)
        lines = [(feature_names_s[j], X_agg_s[i, j], shap_agg_s[i, j].round(5)) for j in
                 range(n_max_features)]
        return lines, np.sum(shap_agg_s[i, :]).round(5)

    def save_shap_force(self, sample_name: str, out_file_path: Union[str, Path],
                        n_max_features: int = 50):
        """
        Save List of tuples containing features, their value in the given sample
        and the resulting change in shap value.
        The list is sorted by the influence of the feature on the shap values of the given sample.

        :param sample_name:
        :param out_file_path:
        :param n_max_features:
        :return:
        """
        assert not Path(str(out_file_path)).is_file(), 'The output file already exists.'
        lines = ['\t'.join([str(y) for y in x]) for x in
                 self.get_shap_force(sample_name=sample_name, n_max_features=n_max_features)[0]]
        with open(out_file_path, 'w') as outf:
            outf.write('feature\tvalue\tshap_value\n')
            outf.write('\n'.join(lines))
            outf.write('\n')

    def save_shap_summary(self, out_file_path: Union[str, Path], n_max_features: int = 50):
        """
        Save summary of features for all predictions, sorted by average impact of feature on shap value.

        :param out_file_path:
        :param n_max_features:
        :return:
        """
        out_file_path = Path(str(out_file_path))
        assert not out_file_path.is_file(), 'The output file already exists.'
        X_agg_s, shap_agg_s, feature_names_s = self._get_sorted_by_shap_data()

        lines = ['feature\tmean_shap_present\tmean_shap_absent\tn_present\tn_absent']
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
            lines.append(
                f'{feature_name}\t{mean_sv_present.round(5)}\t{mean_sv_absent.round(5)}'
                f'\t{n_where_present}\t{n_where_absent}')
        to_write = '\n'.join(lines) + '\n'
        with open(out_file_path, 'w') as outf:
            outf.write(to_write)
