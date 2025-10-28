import os
import pickle
import torch


class BaseCache:
    """
    Base cache that safely stores objects on CPU and can persist to disk.
    Includes automatic GPU tensor cleanup and load/save diagnostics.
    """

    def __init__(self, name, cache_dir=None, persist=False):
        self._cache = {}
        self.name = name
        self.cache_dir = cache_dir
        self.persist = persist

        if cache_dir and persist:
            os.makedirs(cache_dir, exist_ok=True)
            self.load()
            self._purge_cuda_tensors()

    # --------------------------- utils ---------------------------

    @property
    def path(self):
        return os.path.join(self.cache_dir, f"{self.name}.pkl") if self.cache_dir else None

    def _purge_cuda_tensors(self):
        """Detect and move any leftover CUDA tensors to CPU after loading."""
        n_purged = 0
        for k, v in list(self._cache.items()):
            if torch.is_tensor(v) and v.is_cuda:
                self._cache[k] = v.detach().cpu()
                n_purged += 1
        if n_purged > 0:
            print(f"[cache] ⚠️  {self.name}: moved {n_purged} CUDA tensors to CPU to free VRAM.")

    # --------------------------- core ops ---------------------------

    def load(self):
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    self._cache = pickle.load(f)
                print(f"[cache] Loaded {self.name} ({len(self._cache)} entries) from {self.path}")
            except Exception as e:
                print(f"[cache] Failed to load {self.name}: {e}")

    def save(self):
        if not (self.cache_dir and self.persist):
            return
        try:
            with open(self.path, "wb") as f:
                pickle.dump(self._cache, f)
            print(f"[cache] Saved {self.name} ({len(self._cache)} entries) → {self.path}")
        except Exception as e:
            print(f"[cache] Save failed for {self.name}: {e}")

    def get(self, key):
        return self._cache.get(key, None)

    def set(self, key, val):
        """Store safely on CPU."""
        if torch.is_tensor(val):
            val = val.detach().cpu()
        self._cache[key] = val


# --------------------------- concrete caches ---------------------------

class ActivationCache(BaseCache):
    def __init__(self, cache_dir=None, persist=False):
        super().__init__("activations", cache_dir, persist)


class EncodingCache(BaseCache):
    def __init__(self, cache_dir=None, persist=False):
        super().__init__("encodings", cache_dir, persist)


class DecodeCache(BaseCache):
    def __init__(self, cache_dir=None, persist=False):
        super().__init__("decodes", cache_dir, persist)

    def key(self, seq, m_bin, mode):
        return (seq, float(m_bin), mode)

    def get(self, seq, m_bin, mode):
        return super().get(self.key(seq, m_bin, mode))

    def set(self, seq, m_bin, mode, val):
        if torch.is_tensor(val):
            val = val.detach().cpu()
        # decoded sequences should be plain strings if possible
        if isinstance(val, (bytes, bytearray)):
            val = val.decode()
        self._cache[self.key(seq, m_bin, mode)] = val
