"use client";

export function SkeletonCard() {
  return (
    <div className="card animate-pulse">
      <div className="flex items-start gap-3">
        <div className="w-5 h-5 rounded bg-gray-200 dark:bg-gray-700 mt-1 shrink-0" />
        <div className="flex-1 space-y-3">
          <div className="h-5 rounded bg-gray-200 dark:bg-gray-700 w-3/4" />
          <div className="h-4 rounded bg-gray-100 dark:bg-gray-700/50 w-1/3" />
          <div className="flex gap-2">
            <div className="h-6 w-16 rounded-full bg-gray-100 dark:bg-gray-700/50" />
            <div className="h-6 w-12 rounded-full bg-gray-100 dark:bg-gray-700/50" />
            <div className="h-6 w-20 rounded-full bg-gray-100 dark:bg-gray-700/50" />
          </div>
        </div>
        <div className="flex flex-col gap-2 shrink-0 items-end">
          <div className="h-6 w-16 rounded-full bg-gray-200 dark:bg-gray-700" />
          <div className="h-6 w-12 rounded-full bg-gray-200 dark:bg-gray-700" />
        </div>
      </div>
    </div>
  );
}

export function SkeletonGrid({ count = 6 }: { count?: number }) {
  return (
    <div className="space-y-4">
      {Array.from({ length: count }).map((_, i) => (
        <SkeletonCard key={i} />
      ))}
    </div>
  );
}

export function SkeletonInfoBar() {
  return (
    <div className="flex items-center justify-between py-3 px-1 animate-pulse">
      <div className="flex items-center gap-3">
        <div className="h-4 w-32 rounded bg-gray-200 dark:bg-gray-700" />
        <div className="h-4 w-20 rounded bg-gray-200 dark:bg-gray-700" />
      </div>
      <div className="flex items-center gap-1">
        <div className="h-8 w-8 rounded bg-gray-200 dark:bg-gray-700" />
        <div className="h-8 w-8 rounded bg-gray-200 dark:bg-gray-700" />
        <div className="flex gap-0.5 mx-1">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="h-8 w-8 rounded bg-gray-100 dark:bg-gray-700/50" />
          ))}
        </div>
        <div className="h-8 w-8 rounded bg-gray-200 dark:bg-gray-700" />
        <div className="h-8 w-8 rounded bg-gray-200 dark:bg-gray-700" />
      </div>
    </div>
  );
}