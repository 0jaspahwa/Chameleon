import React from 'react';

import { Navbar } from './Navbar';
import { Footer } from './Footer';

interface PageLayoutProps {
  children: React.ReactNode;
}

export function PageLayout({ children }: PageLayoutProps) {

  return (
    <div className="app-background min-h-screen flex flex-col">

      {/* Shared Navbar */}
      <Navbar title="UltraPersonal">

        <a
          href="#"
          className="text-sm text-[#666] hover:text-[#111] transition-colors font-inter"
        >
          Shop
        </a>

        <a
          href="#"
          className="text-sm text-[#666] hover:text-[#111] transition-colors font-inter"
        >
          Collections
        </a>

        <a
          href="#"
          className="text-sm text-[#666] hover:text-[#111] transition-colors font-inter"
        >
          About
        </a>

      </Navbar>

      {/* Main Content */}
      <main className="flex-grow">
        {children}
      </main>

      {/* Shared Footer */}
      <Footer brandName="UltraPersonal" />

    </div>
  );
}