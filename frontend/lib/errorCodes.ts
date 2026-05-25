export const ERROR_CODE_MAP: Record<number, string> = {
  400: "请求参数有误，请检查输入",
  401: "登录已过期，请重新登录",
  403: "无权限访问该资源",
  404: "请求的资源不存在或已被删除",
  413: "文件大小超过限制，请压缩后上传",
  429: "操作太频繁，请稍后再试",
  500: "服务器出现异常，已通知管理员",
  502: "服务暂时不可用，请稍后重试",
  503: "服务正在维护中，请稍后重试",
};

export function getErrorMessage(status: number, fallback?: string): string {
  return ERROR_CODE_MAP[status] || fallback || `请求失败 (${status})`;
}

export function getErrorHint(status: number): string | null {
  const hints: Record<number, string> = {
    401: "请重新登录以继续操作",
    403: "请检查您的账户权限设置",
    404: "该知识可能已被其他用户删除",
    413: "建议将文件压缩至 50MB 以内",
    429: "每分钟操作次数有限制，请稍等片刻",
    500: "我们已收到异常报告，请稍后重新尝试",
  };
  return hints[status] || null;
}