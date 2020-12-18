import importlib_resources
import random
import os, urllib, zipfile, tarfile, requests
from pathlib import Path
from functools import lru_cache
from typing import Union, Callable, Dict
import torch
import tabulate


def load_resource(path: str):
    return importlib_resources.files("fastaug") / "resources" / path


def randint(a, b):
    return random.randint(a, b - 1)


def cache_dir(dest_dir):
    return Path(os.environ['HOME']).expanduser() / ".cache" / "fastaug" / dest_dir


class DownloadUtil:
    @staticmethod
    def download_counter_fitting_if_not_exists():
        if not cache_dir("counter-fitting").exists():
            url = 'https://raw.githubusercontent.com/nmrksic/counter-fitting/master/word_vectors/glove.txt.zip'
            print(f"Downloading from {url}")
            file_path = DownloadUtil.download(
                url, dest_dir=str(cache_dir("counter-fitting")))
            print("Unzip...")
            DownloadUtil.unzip(file_path)
        else:
            print("File exists...")
        return cache_dir("counter-fitting") / "glove.txt"

    @staticmethod
    def download(src, dest_dir, dest_file=None):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        if dest_file is None:
            dest_file = os.path.basename(src)

        if not os.path.exists(dest_dir + dest_file):
            req = urllib.request.Request(src)
            file = urllib.request.urlopen(req)
            with open(os.path.join(dest_dir, dest_file), 'wb') as output:
                output.write(file.read())
        return os.path.join(dest_dir, dest_file)

    @staticmethod
    def unzip(file_path, dest_dir=None):
        """
        :param str file_path: File path for unzip
        >>> DownloadUtil.unzip('zip_file.zip')
        """

        if dest_dir is None:
            dest_dir = os.path.dirname(file_path)

        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(dest_dir)
        elif file_path.endswith("tar.gz") or file_path.endswith("tgz"):
            tar = tarfile.open(file_path, "r:gz")
            tar.extractall(dest_dir)
            tar.close()
        elif file_path.endswith("tar"):
            tar = tarfile.open(file_path, "r:")
            tar.extractall(dest_dir)
            tar.close()


class EmbeddingNbrUtil:
    """
        When loading a pretrained embedding matrix, some words may not
        occur in the pretrained vectors, thus they may be filled with random
        numbers. This may cause strange results when querying neariest 
        neighbours. We assume that all words that are not found in the 
        pretrained vectors will be filled with 0. Querying these words will
        return an empty list.
    """
    def __init__(
        self,
        embed: torch.Tensor,
        word2idx: Union[Callable, Dict],
        idx2word: Union[Callable, Dict],
    ):
        self.embed = embed
        if isinstance(word2idx, dict):
            self.word2idx = word2idx.__getitem__
        else:
            self.word2idx = word2idx
        if isinstance(idx2word, dict):
            self.idx2word = idx2word.__getitem__
        else:
            self.idx2word = idx2word
        self._cache = {}

    def is_pretrained(self, element: Union[int, str]):
        return not all(self.as_vector(element) == 0.0)

    def as_vector(self, element: Union[int, str, torch.Tensor]):
        if isinstance(element, int):
            idx = element
            query_vector = self.embed[idx]
        elif isinstance(element, str):
            idx = self.word2idx(element)
            query_vector = self.embed[idx]
        elif isinstance(element, torch.Tensor):
            query_vector = element
        else:
            raise TypeError(
                'You passed a {}, int/str/torch.Tensor required'.format(
                    type(element)))
        return query_vector

    def as_index(self, element: Union[int, str]):
        if isinstance(element, int):
            idx = element
        elif isinstance(element, str):
            idx = self.word2idx(element)
        else:
            raise TypeError('You passed a {}, int/str required'.format(
                type(element)))
        return idx

    # search neighbours of all words and save them into a cache,
    # this will speed up the query process.
    # The pre_search is rather fast, feel free to use it.
    def pre_search(self, measure='euc', topk=None, gpu=True):
        try:
            import faiss
        except Exception as e:
            print("please install faiss first!")
            return
        data = self.embed.cpu().numpy()
        dim = self.embed.size(1)
        index = faiss.IndexFlatL2(dim)
        index.add(data)
        if gpu:
            res = faiss.StandardGpuResources()  # use a single GPU
            index = faiss.index_cpu_to_gpu(res, 0, index)
        D, I = index.search(data, topk)
        self._cache[f'D-{measure}-{topk}'] = D
        self._cache[f'I-{measure}-{topk}'] = I


    @lru_cache(maxsize=None)
    @torch.no_grad()
    def find_neighbours(
            self,
            element: Union[int, str, torch.Tensor],
            measure='euc',
            topk=None,
            dist=None,
            return_words=False,  # by default, return (D, I)
            ):
        # checking args
        assert measure in ['euc', 'cos']
        if dist is not None:
            assert (measure == 'euc' and dist > 0) or (
                measure == 'cos' and 0 < dist < 1
            ), "threshold for euc distance must be larger than 0, for cos similarity must be between 0 and 1"

        measure_fn = cos_dist if measure == 'cos' else euc_dist
        query_vector = self.as_vector(element)

        # Assume that a vector equals to 0 has no neighbours
        if not self.is_pretrained(query_vector):
            if return_words:
                return []
            else:
                return None, None

        if f'D-{measure}-{topk}' in self._cache:
            _idx = self.as_index(element)
            D = self._cache[f'D-{measure}-{topk}'][_idx]
            I = self._cache[f'I-{measure}-{topk}'][_idx]
            tk_vals = torch.tensor(D, device=self.embed.device)
            tk_idxs = torch.tensor(I, device=self.embed.device)
            if return_words:
                return [self.idx2word(ele) for ele in tk_idxs.tolist()]
            else:
                return tk_vals, tk_idxs

        if topk is None:
            _topk = self.embed.size(0)
        else:
            _topk = topk

        dists = measure_fn(query_vector, self.embed)
        tk_vals, tk_idxs = torch.topk(dists, _topk, largest=False)

        if dist is not None:
            mask_idx = tk_vals < dist
            tk_vals = tk_vals[mask_idx]
            tk_idxs = tk_idxs[mask_idx]

        if return_words:
            return [self.idx2word(ele) for ele in tk_idxs.tolist()]
        else:
            return tk_vals, tk_idxs


# torch.tensor([100]).expand(2000, 100):
#   return a view, in memory it's still [100]
# torch.tensor([100]).expand(2000, 1):
#   return a copied tensor
def cos_dist(qry, mem):
    return 1 - cos_sim(qry, mem)


def cos_sim(qry, mem):
    return torch.nn.functional.cosine_similarity(mem,
                                                 qry.expand(
                                                     mem.size(0), mem.size(1)),
                                                 dim=1)


def euc_dist(qry, mem):
    return torch.sqrt((qry - mem).pow(2).sum(dim=1))
