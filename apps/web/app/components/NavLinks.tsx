"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function NavLinks() {
  const pathname = usePathname();

  const links = [
    { href: "/lab", label: "Lab" },
    { href: "/learn", label: "Learn" },
    { href: "/playground", label: "Playground" },
    { href: "/auction", label: "Auction" },
    { href: "/studio", label: "Studio" },
    { href: "/models", label: "Models" },
    { href: "/findings", label: "Findings" },
  ];

  const isActive = (href: string) => {
    if (href === "/") return pathname === "/";
    return pathname.startsWith(href);
  };

  return (
    <div className="links">
      {links.map((link) => (
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
