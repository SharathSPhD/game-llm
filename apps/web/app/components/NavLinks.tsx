"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function NavLinks() {
  const pathname = usePathname();

  const links = [
    { href: "/", label: "Home" },
    { href: "/lab", label: "Lab" },
    { href: "/qre", label: "QRE" },
    { href: "/auction", label: "Auction" },
    { href: "/playground", label: "Playground" },
    { href: "/studio", label: "Studio" },
    { href: "/models", label: "Models" },
    { href: "/findings", label: "Findings" },
  ];

  return (
    <nav className="nav-links">
      {links.map((link) => (
        <Link
          key={link.href}
          href={link.href}
          className={`nav-link ${pathname === link.href ? "active" : ""}`}
        >
          {link.label}
        </Link>
      ))}
    </nav>
  );
}
