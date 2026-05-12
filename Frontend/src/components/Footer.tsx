import React from 'react';
import { Link } from 'react-router-dom';

interface FooterProps {
  brandName?: string;
}

export function Footer({
  brandName = 'Ultra Personalization Engine',
}: FooterProps) {

  return (
    <footer className="w-full mt-20">

      {/* Top Border */}
      <div className="border-t border-[#e5e5e1]" />

      {/* Footer Content */}
      <div className="max-w-7xl mx-auto px-6 md:px-12 py-8 flex flex-col items-center">

        {/* Links */}
        <div className="flex flex-wrap items-center justify-center gap-10 text-[15px] tracking-[0.08em] text-[#8b8b87] font-inter uppercase">

          <Link
            to="#"
            className="hover:text-[#1d1d1d] transition-colors duration-300"
          >
            About
          </Link>

          <Link
            to="#"
            className="hover:text-[#1d1d1d] transition-colors duration-300"
          >
            Journal
          </Link>

          <Link
            to="#"
            className="hover:text-[#1d1d1d] transition-colors duration-300"
          >
            Contact
          </Link>

          <Link
            to="#"
            className="hover:text-[#1d1d1d] transition-colors duration-300"
          >
            Terms
          </Link>

        </div>

        {/* Copyright */}
        <p className="mt-6 text-[15px] text-[#b0b0ab] text-center font-inter">

          © 2026 {brandName}. Built for the next generation of commerce.

        </p>

      </div>

    </footer>
  );
}