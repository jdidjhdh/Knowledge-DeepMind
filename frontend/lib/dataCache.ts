type CacheEntry<T> = { data: T; expiry: number; staledAt: number };

const cache = new Map<string, CacheEntry<any>>();
const pending = new Map<string, Promise<any>>();

const DEFAULT_TTL_MS = 60_000;
const SWR_GRACE_MS = 120_000;

function cacheGet<T>(key: string): { data: T; stale: boolean } | null {
  const entry = cache.get(key);
  if (!entry) return null;
  if (entry.expiry > Date.now()) return { data: entry.data, stale: false };
  if (entry.staledAt > Date.now()) return { data: entry.data, stale: true };
  cache.delete(key);
  return null;
}

function cacheSet<T>(key: string, data: T, ttlMs = DEFAULT_TTL_MS) {
  const now = Date.now();
  cache.set(key, { data, expiry: now + ttlMs, staledAt: now + ttlMs + SWR_GRACE_MS });
}

async function cachedFetch<T>(
  key: string,
  fetcher: () => Promise<T>,
  ttlMs = DEFAULT_TTL_MS
): Promise<T> {
  const cached = cacheGet<T>(key);
  if (cached && !cached.stale) return cached.data;

  if (cached && cached.stale) {
    fetcher()
      .then((data) => cacheSet(key, data, ttlMs))
      .catch(() => {});
    return cached.data;
  }

  if (pending.has(key)) return pending.get(key) as Promise<T>;

  const promise = fetcher()
    .then((data) => {
      cacheSet(key, data, ttlMs);
      pending.delete(key);
      return data;
    })
    .catch((err) => {
      pending.delete(key);
      throw err;
    });

  pending.set(key, promise);
  return promise;
}

function invalidateCache(patterns?: string | string[]) {
  if (!patterns) {
    cache.clear();
    return;
  }
  const patternList = Array.isArray(patterns) ? patterns : [patterns];
  for (const pattern of patternList) {
    for (const key of Array.from(cache.keys())) {
      if (key.includes(pattern)) cache.delete(key);
    }
  }
}

function getCacheStats() {
  const now = Date.now();
  let fresh = 0;
  let stale = 0;
  cache.forEach((entry) => {
    if (entry.expiry > now) fresh++;
    else if (entry.staledAt > now) stale++;
  });
  return { total: cache.size, fresh, stale, pending: pending.size };
}

function clearCache() {
  cache.clear();
  pending.clear();
}

export { cachedFetch, invalidateCache, cacheSet, cacheGet, getCacheStats, clearCache };