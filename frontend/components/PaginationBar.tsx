"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from "lucide-react";

interface PaginationBarProps {
  page: number;
  pageSize: number;
  total: number;
  totalPages: number;
  hasPrev: boolean;
  hasNext: boolean;
  onPageChange: (page: number) => void;
  onPageSizeChange?: (size: number) => void;
  loading?: boolean;
}

export default function PaginationBar({
  page,
  pageSize,
  total,
  totalPages,
  hasPrev,
  hasNext,
  onPageChange,
  onPageSizeChange,
  loading = false,
}: PaginationBarProps) {
  const [jumpInput, setJumpInput] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setJumpInput("");
  }, [page]);

  const handleJump = () => {
    const target = parseInt(jumpInput, 10);
    if (isNaN(target) || target < 1 || target > totalPages) return;
    onPageChange(target);
  };

  const renderPageButtons = useCallback(() => {
    if (totalPages <= 7) {
      return Array.from({ length: totalPages }, (_, i) => i + 1).map((p) => (
        <PageButton key={p} num={p} active={p === page} onClick={() => onPageChange(p)} />
      ));
    }

    const buttons: React.ReactNode[] = [];
    buttons.push(<PageButton key={1} num={1} active={page === 1} onClick={() => onPageChange(1)} />);

    let left = Math.max(2, page - 1);
    let right = Math.min(totalPages - 1, page + 1);
    if (page <= 3) right = Math.min(5, totalPages - 1);
    if (page >= totalPages - 2) left = Math.max(totalPages - 4, 2);

    if (left > 2) buttons.push(<Ellipsis key="el1" />);
    for (let p = left; p <= right; p++) {
      buttons.push(<PageButton key={p} num={p} active={p === page} onClick={() => onPageChange(p)} />);
    }
    if (right < totalPages - 1) buttons.push(<Ellipsis key="el2" />);

    if (totalPages > 1) {
      buttons.push(
        <PageButton key={totalPages} num={totalPages} active={page === totalPages} onClick={() => onPageChange(totalPages)} />
      );
    }
    return buttons;
  }, [page, totalPages, onPageChange]);

  if (total === 0 && !loading) return null;

  return (
    <div className="flex items-center justify-between py-3 px-1">
      <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
        <span>
          共 <strong className="text-gray-700 dark:text-gray-200">{total}</strong> 条
        </span>
        <span>·</span>
        <span>
          第 <strong className="text-gray-700 dark:text-gray-200">{page}/{totalPages}</strong> 页
        </span>
        {onPageSizeChange && (
          <select
            value={pageSize}
            onChange={(e) => onPageSizeChange(Number(e.target.value))}
            className="ml-2 text-xs rounded border border-gray-200 dark:border-gray-600 bg-transparent px-2 py-0.5"
          >
            {[10, 20, 50, 100].map((s) => (
              <option key={s} value={s}>{s}条/页</option>
            ))}
          </select>
        )}
      </div>

      {totalPages > 1 && (
        <div className="flex items-center gap-1">
          <NavButton
            onClick={() => onPageChange(1)}
            disabled={!hasPrev || loading}
            title="首页"
          >
            <ChevronsLeft className="w-4 h-4" />
          </NavButton>
          <NavButton
            onClick={() => onPageChange(page - 1)}
            disabled={!hasPrev || loading}
            title="上一页"
          >
            <ChevronLeft className="w-4 h-4" />
          </NavButton>

          <div className="hidden sm:flex items-center gap-0.5 mx-1">
            {renderPageButtons()}
          </div>

          <span className="sm:hidden text-xs text-gray-500 mx-1">
            {page}/{totalPages}
          </span>

          <NavButton
            onClick={() => onPageChange(page + 1)}
            disabled={!hasNext || loading}
            title="下一页"
          >
            <ChevronRight className="w-4 h-4" />
          </NavButton>
          <NavButton
            onClick={() => onPageChange(totalPages)}
            disabled={!hasNext || loading}
            title="末页"
          >
            <ChevronsRight className="w-4 h-4" />
          </NavButton>

          {totalPages > 10 && (
            <div className="hidden sm:flex items-center gap-1 ml-2">
              <input
                ref={inputRef}
                type="text"
                value={jumpInput}
                onChange={(e) => setJumpInput(e.target.value.replace(/[^0-9]/g, ""))}
                onKeyDown={(e) => { if (e.key === "Enter") handleJump(); }}
                placeholder="页码"
                className="w-12 text-xs text-center rounded border border-gray-200 dark:border-gray-600 bg-transparent py-1"
                disabled={loading}
              />
              <button
                onClick={handleJump}
                disabled={!jumpInput || loading}
                className="text-xs px-2 py-1 rounded bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 disabled:opacity-30 transition-colors"
              >
                跳转
              </button>
            </div>
          )}
        </div>
      )}

      {loading && (
        <div className="absolute inset-0 bg-white/50 dark:bg-gray-900/30 rounded flex items-center justify-center" />
      )}
    </div>
  );
}

function PageButton({ num, active, onClick }: { num: number; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`w-8 h-8 text-sm rounded transition-colors ${
        active
          ? "bg-primary-600 text-white font-semibold"
          : "text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
      }`}
    >
      {num}
    </button>
  );
}

function Ellipsis() {
  return <span className="w-8 h-8 flex items-center justify-center text-gray-400">…</span>;
}

function NavButton({
  onClick,
  disabled,
  title,
  children,
}: {
  onClick: () => void;
  disabled: boolean;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className="p-1.5 rounded text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-25 disabled:cursor-not-allowed transition-colors"
    >
      {children}
    </button>
  );
}