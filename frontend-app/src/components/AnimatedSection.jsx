import { useEffect, useRef } from "react";

/**
 * Wraps children and reveals them with animation when they scroll into view.
 * Uses IntersectionObserver — no scroll event listeners.
 */
function AnimatedSection({
  children,
  className = "",
  animation = "reveal",  // "reveal" | "reveal-scale" | "reveal-left" | "reveal-right" | "stagger-children"
  delay = 0,
  threshold = 0.15,
  tag: Tag = "div",
}) {
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    // Respect reduced motion
    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    if (prefersReducedMotion) {
      el.classList.add("visible");
      return;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          if (delay > 0) {
            setTimeout(() => el.classList.add("visible"), delay);
          } else {
            el.classList.add("visible");
          }
          observer.unobserve(el);
        }
      },
      { threshold }
    );

    observer.observe(el);

    return () => observer.disconnect();
  }, [delay, threshold]);

  return (
    <Tag ref={ref} className={`${animation} ${className}`}>
      {children}
    </Tag>
  );
}

export default AnimatedSection;
