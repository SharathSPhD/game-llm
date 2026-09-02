"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function NavLinks() {
  const pathname = usePathname();

  // Primary navigation: the 4 key pages
  const primaryLinks = [
    { href: "/benchmarks", label: "Benchmarks" },
    { href: "/api", label: "API" },
    { href: "/demo", label: "Demo" },
  ];

  // Secondary tools
  const toolLinks = [
    { href: "/lab", label: "Lab" },
    { href: "/leaderboard", label: "Leaderboard" },
    { href: "/studio", label: "Registry" },
    { href: "/findings", label: "Findings" },
  ];

  const isActive = (href: string) => {
    if (href === "/") return pathname === "/";
    return pathname.startsWith(href);
  };

  return (
    <div className="links">
      {primaryLinks.map((link) => (
        <Link
          key={link.href}
          href={link.href}
          data-active={isActive(link.href) ? "true" : "false"}
        >
          {link.label}
        </Link>
      ))}
      <div style={{ borderRight: "1px solid var(--border-subtle)" }} />
      {toolLinks.map((link) => (
        <Link
          key={link.href}
          href={link.href}
          data-active={isActive(link.href) ? "true" : "false"}
        >
          {link.label}
        </Link>
      ))}
    </div>
  );
}
