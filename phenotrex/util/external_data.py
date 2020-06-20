import io
import gzip
from typing import Dict, Tuple
import urllib.request
from urllib.error import HTTPError

from phenotrex.util.logging import get_logger


class Eggnog5TextAnnotator:

    BASE_PATH = "http://eggnog5.embl.de/download/eggnog_5.0/per_tax_level"

    def __init__(self):
        self._known_taxa = {}
        self.logger = get_logger(self.__class__.__name__)

    @staticmethod
    def _download_and_zcat(http_path: str) -> str:
        try:
            response = urllib.request.urlopen(http_path, timeout=5)
            gz = io.BytesIO(response.read())
            with gzip.open(gz, mode='r') as gzf:
                return gzf.read().decode('utf8')
        except HTTPError:
            return ''

    def _load_taxon(self, taxon_id: int):
        dlp = f'{Eggnog5TextAnnotator.BASE_PATH}/{str(taxon_id)}/{taxon_id}_annotations.tsv.gz'
        loaded = [x.strip().split('\t') for x in self._download_and_zcat(str(dlp)).split('\n')]
        loaded = [x for x in loaded if len(x) == 4]
        loaded = {x: {'type': y, 'annotation': z} for _, x, y, z in loaded}
        if not loaded:
            self.logger.warning(f'Could not load annotations for taxon {taxon_id} from eggNOG5.')
        else:
            self._known_taxa[taxon_id] = loaded

    def _get_taxon_annotations(self, taxon_id: int) -> Dict[str, Dict[str, str]]:
        if taxon_id not in self._known_taxa:
            self._load_taxon(taxon_id)
        return self._known_taxa.get(taxon_id, {})

    def annotate(self, taxon_id: int, enog_id: str) -> Tuple[str, str]:
        """
        Load text annotations for the given enog_id from cached eggNOG5 annotations;
        if no annotations are cached for the given Taxon ID, download them.

        :param taxon_id: The NCBI taxon ID at which the eggNOG cluster is situated.
        :param enog_id: The eggNOG5 ID for which annotations are to be loaded.
        :returns: the enog type and annotation (if one exists, else empty strings).
        """
        d = self._get_taxon_annotations(taxon_id).get(enog_id)
        if d:
            return d['type'], d['annotation']
        else:
            return '', ''
