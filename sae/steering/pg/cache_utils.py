import os, pickle

class BaseCache:
    def __init__(self, name, cache_dir=None, persist=False):
        self._cache = {}
        self.name = name
        self.cache_dir = cache_dir
        self.persist = persist
        if cache_dir and persist:
            os.makedirs(cache_dir, exist_ok=True)
            self.load()

    @property
    def path(self):
        if not self.cache_dir: return None
        return os.path.join(self.cache_dir, f"{self.name}.pkl")

    def load(self):
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    self._cache = pickle.load(f)
                print(f"[cache] Loaded {self.name} ({len(self._cache)} entries)")
            except Exception as e:
                print(f"[cache] Failed to load {self.name}: {e}")

    def save(self):
        if self.path and self.persist:
            with open(self.path, "wb") as f:
                pickle.dump(self._cache, f)
            print(f"[cache] Saved {self.name} ({len(self._cache)} entries)")

    def get(self, key):
        return self._cache.get(key, None)

    def set(self, key, val):
        self._cache[key] = val


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
        super().set(self.key(seq, m_bin, mode), val)
