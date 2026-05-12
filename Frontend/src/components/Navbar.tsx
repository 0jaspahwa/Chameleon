import React from 'react';

interface NavbarProps {
  title?: string;
  transparent?: boolean;
  showCart?: boolean;
  children?: React.ReactNode;
}

export function Navbar({
  title = "UltraPersonal",
  transparent = false,
  showCart = true,
  children,
}: NavbarProps) {

  return (
    <header
      className={`
        flex items-center justify-between
        whitespace-nowrap
        px-10 py-3
        sticky top-0 z-50
        backdrop-blur-md
        rounded-b-lg
        transition-all duration-300
        ${transparent
          ? 'bg-white/40'
          : 'bg-white/90 shadow-sm border-b border-black/5'}
      `}
    >

      {/* Left */}
      <div className="flex items-center gap-4 text-[#1d1d1d]">

        <div className="size-4">
          <svg
            fill="none"
            viewBox="0 0 48 48"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M44 4H30.6666V17.3334H17.3334V30.6666H4V44H44V4Z"
              fill="currentColor"
            />
          </svg>
        </div>

        <h2 className="font-jakarta text-lg font-medium tracking-[-0.015em]">
          {title}
        </h2>

      </div>

      {/* Center / Custom Content */}
      {children && (
        <div className="hidden md:flex items-center gap-8">
          {children}
        </div>
      )}

      {/* Right */}
      {showCart && (
        <button className="flex items-center justify-center rounded-full h-10 w-10 hover:bg-black/5 text-[#1d1d1d] transition-colors">

          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            >
            <path
                d="M4 7H20"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
            />

            <path
                d="M7 12H17"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
            />

            <path
                d="M10 17H14"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
            />
            </svg>

        </button>
      )}

    </header>
  );
}