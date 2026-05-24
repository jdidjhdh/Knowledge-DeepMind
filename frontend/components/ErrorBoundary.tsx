"use client";

import { Component, ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error) {
    console.error("[ErrorBoundary]", error);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;

      return (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <AlertTriangle className="w-16 h-16 text-orange-500 mb-4" />
          <h2 className="text-xl font-semibold mb-2">页面渲染出错</h2>
          <p className="text-gray-500 dark:text-gray-400 mb-6 max-w-md">
            {this.state.error?.message || "未知错误"}
          </p>
          <button
            onClick={this.handleRetry}
            className="btn-primary flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            重试
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}