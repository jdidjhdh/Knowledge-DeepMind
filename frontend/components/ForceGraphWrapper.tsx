"use client";

import { forwardRef, useEffect, useState, ComponentType, memo, useRef } from "react";

const ForceGraphWrapper = memo(
  forwardRef<any, any>((props, ref) => {
    const [Comp, setComp] = useState<ComponentType<any> | null>(null);
    const compRef = useRef<ComponentType<any> | null>(null);

    useEffect(() => {
      let cancelled = false;
      import("react-force-graph-2d").then((mod) => {
        if (!cancelled) {
          compRef.current = mod.default;
          setComp(() => mod.default);
        }
      });
      return () => {
        cancelled = true;
      };
    }, []);

    const C = Comp || compRef.current;

    if (!C) {
      return <div style={{ width: props.width || "100%", height: props.height || 600 }} />;
    }

    const { graphData: _gd, ...rest } = props;
    return <C ref={ref} graphData={_gd} {...rest} />;
  }),
  (prevProps, nextProps) => {
    const prevKeys = Object.keys(prevProps);
    const nextKeys = Object.keys(nextProps);
    if (prevKeys.length !== nextKeys.length) return false;

    for (const key of prevKeys) {
      if (key === "graphData") {
        const pn = prevProps.graphData?.nodes?.length ?? 0;
        const nn = nextProps.graphData?.nodes?.length ?? 0;
        const pe = prevProps.graphData?.links?.length ?? 0;
        const ne = nextProps.graphData?.links?.length ?? 0;
        if (pn !== nn || pe !== ne) return false;
        continue;
      }
      if (key === "ref") continue;
      if (prevProps[key] !== nextProps[key]) return false;
    }
    return true;
  },
);

ForceGraphWrapper.displayName = "ForceGraphWrapper";

export default ForceGraphWrapper;