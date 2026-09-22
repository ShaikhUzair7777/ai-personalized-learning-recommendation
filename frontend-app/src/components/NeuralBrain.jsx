import { useRef, useEffect, useCallback } from "react";

/**
 * Canvas-based neural brain animation.
 * Pure Canvas API — no dependencies. GPU-friendly (transform/opacity only).
 *
 * Props:
 *  - size: canvas dimension (default 400)
 *  - intensity: 0-1 controlling glow/pulse (default 1)
 *  - animate: whether to run animation loop (default true)
 *  - className: optional wrapper class
 */

// Brain node positions (pre-computed, normalized 0-1)
const BRAIN_NODES = [
  // Left hemisphere outline
  { x: 0.28, y: 0.18 }, { x: 0.22, y: 0.28 }, { x: 0.18, y: 0.40 },
  { x: 0.17, y: 0.52 }, { x: 0.20, y: 0.63 }, { x: 0.25, y: 0.72 },
  { x: 0.32, y: 0.78 }, { x: 0.40, y: 0.82 },
  // Right hemisphere outline
  { x: 0.72, y: 0.18 }, { x: 0.78, y: 0.28 }, { x: 0.82, y: 0.40 },
  { x: 0.83, y: 0.52 }, { x: 0.80, y: 0.63 }, { x: 0.75, y: 0.72 },
  { x: 0.68, y: 0.78 }, { x: 0.60, y: 0.82 },
  // Top center
  { x: 0.42, y: 0.14 }, { x: 0.50, y: 0.12 }, { x: 0.58, y: 0.14 },
  // Internal nodes
  { x: 0.35, y: 0.30 }, { x: 0.45, y: 0.25 }, { x: 0.55, y: 0.25 },
  { x: 0.65, y: 0.30 }, { x: 0.30, y: 0.45 }, { x: 0.40, y: 0.38 },
  { x: 0.50, y: 0.35 }, { x: 0.60, y: 0.38 }, { x: 0.70, y: 0.45 },
  { x: 0.35, y: 0.55 }, { x: 0.45, y: 0.50 }, { x: 0.55, y: 0.50 },
  { x: 0.65, y: 0.55 }, { x: 0.38, y: 0.65 }, { x: 0.50, y: 0.60 },
  { x: 0.62, y: 0.65 }, { x: 0.45, y: 0.72 }, { x: 0.55, y: 0.72 },
  // Brain stem
  { x: 0.48, y: 0.86 }, { x: 0.52, y: 0.86 }, { x: 0.50, y: 0.90 },
];

// Connections between nodes (index pairs)
const BRAIN_CONNECTIONS = [
  // Left outline
  [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
  // Right outline
  [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
  // Top
  [0, 16], [16, 17], [17, 18], [18, 8],
  // Internal cross-connections
  [19, 20], [20, 21], [21, 22], [19, 24], [20, 25], [21, 26], [22, 27],
  [23, 24], [24, 25], [25, 26], [26, 27],
  [23, 28], [28, 29], [29, 30], [30, 27],
  [28, 31], [29, 32], [30, 33],
  [31, 32], [32, 33],
  [31, 34], [33, 35],
  [34, 35],
  // Connect outline to internal
  [0, 19], [1, 23], [2, 23], [3, 28], [4, 31], [5, 34],
  [8, 22], [9, 27], [10, 27], [11, 30], [12, 33], [13, 35],
  [16, 20], [18, 21], [6, 34], [14, 35],
  // Brain stem
  [7, 36], [15, 37], [36, 37], [36, 38], [37, 38],
  [34, 36], [35, 37],
];

function NeuralBrain({
  size = 400,
  intensity = 1,
  animate = true,
  className = "",
}) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const timeRef = useRef(0);

  const draw = useCallback(
    (ctx, width, height, time) => {
      ctx.clearRect(0, 0, width, height);

      const scale = Math.min(width, height);
      const ox = (width - scale) / 2;
      const oy = (height - scale) / 2;

      // Map nodes to pixel positions
      const nodes = BRAIN_NODES.map((n) => ({
        x: ox + n.x * scale,
        y: oy + n.y * scale,
      }));

      // Draw connections
      for (const [a, b] of BRAIN_CONNECTIONS) {
        const na = nodes[a];
        const nb = nodes[b];

        // Pulse effect traveling along connection
        const pulsePhase =
          (time * 0.001 + (a + b) * 0.15) % (Math.PI * 2);
        const pulseAlpha =
          0.08 + 0.12 * intensity * Math.sin(pulsePhase) ** 2;

        ctx.beginPath();
        ctx.moveTo(na.x, na.y);
        ctx.lineTo(nb.x, nb.y);
        ctx.strokeStyle = `rgba(164, 155, 255, ${pulseAlpha})`;
        ctx.lineWidth = 1;
        ctx.stroke();

        // Traveling data pulse (small bright dot along line)
        const t = ((time * 0.0005 + a * 0.3) % 1);
        const px = na.x + (nb.x - na.x) * t;
        const py = na.y + (nb.y - na.y) * t;
        const dotAlpha = 0.3 * intensity * Math.sin(t * Math.PI);
        if (dotAlpha > 0.05) {
          ctx.beginPath();
          ctx.arc(px, py, 1.5, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(164, 155, 255, ${dotAlpha})`;
          ctx.fill();
        }
      }

      // Draw nodes
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        const pulse = Math.sin(time * 0.002 + i * 0.5);
        const nodeRadius = 2.5 + 0.8 * pulse * intensity;
        const alpha = (0.4 + 0.3 * pulse) * intensity;

        // Glow
        ctx.beginPath();
        ctx.arc(n.x, n.y, nodeRadius + 4, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(124, 106, 255, ${alpha * 0.15})`;
        ctx.fill();

        // Node
        ctx.beginPath();
        ctx.arc(n.x, n.y, nodeRadius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(164, 155, 255, ${alpha})`;
        ctx.fill();
      }

      // Center glow
      const cx = ox + 0.5 * scale;
      const cy = oy + 0.5 * scale;
      const glowPulse = 0.5 + 0.5 * Math.sin(time * 0.001);
      const grad = ctx.createRadialGradient(
        cx, cy, 0, cx, cy, scale * 0.25
      );
      grad.addColorStop(0, `rgba(124, 106, 255, ${0.06 * intensity * glowPulse})`);
      grad.addColorStop(1, "rgba(124, 106, 255, 0)");
      ctx.beginPath();
      ctx.arc(cx, cy, scale * 0.25, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();

      // Floating particles around brain
      for (let i = 0; i < 12; i++) {
        const angle = (time * 0.0003 + i * (Math.PI * 2 / 12)) % (Math.PI * 2);
        const orbit = scale * (0.38 + 0.05 * Math.sin(time * 0.001 + i));
        const px = cx + Math.cos(angle) * orbit;
        const py = cy + Math.sin(angle) * orbit;
        const pAlpha = (0.15 + 0.1 * Math.sin(time * 0.002 + i)) * intensity;

        ctx.beginPath();
        ctx.arc(px, py, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(76, 164, 255, ${pAlpha})`;
        ctx.fill();
      }
    },
    [intensity]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    resize();
    window.addEventListener("resize", resize);

    if (!animate) {
      draw(ctx, canvas.width / dpr, canvas.height / dpr, 0);
      return () => window.removeEventListener("resize", resize);
    }

    let running = true;
    const loop = (timestamp) => {
      if (!running) return;
      timeRef.current = timestamp;
      const rect = canvas.getBoundingClientRect();
      draw(ctx, rect.width, rect.height, timestamp);
      animRef.current = requestAnimationFrame(loop);
    };

    animRef.current = requestAnimationFrame(loop);

    return () => {
      running = false;
      if (animRef.current) cancelAnimationFrame(animRef.current);
      window.removeEventListener("resize", resize);
    };
  }, [animate, draw]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{
        width: size,
        height: size,
        display: "block",
      }}
      aria-hidden="true"
    />
  );
}

export default NeuralBrain;
