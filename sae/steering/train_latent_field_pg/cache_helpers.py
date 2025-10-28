from sae.steering.pg.cache_utils import ActivationCache, EncodingCache, DecodeCache
def make_caches(args):
    act = ActivationCache(args.cache_dir, args.persist_caches) if args.cache_activations else None
    enc = EncodingCache(args.cache_dir, args.persist_caches) if args.cache_encodings else None
    dec = DecodeCache(args.cache_dir, args.persist_caches) if args.cache_decoded else None
    return act, enc, dec
def cache_coverage(cache, seqs):
    if cache is None or not hasattr(cache, '_cache'):
        return 0, 0.0
    n_cached = sum(1 for s in seqs if s in cache._cache)
    frac = n_cached / max(1, len(seqs))
    return n_cached, frac
