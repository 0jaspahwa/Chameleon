import React from 'react';

import { Link } from 'react-router-dom';

import { PageLayout } from '../components/PageLayout';
import DynamicWeightText from '../components/Dynamicweighttext';
import chameleonLogo from '../components/chameleon-logo.png';

export function LandingPage() {

  return (
    

      <section className="flex flex-col items-center justify-center min-h-[85vh] px-6 text-center">

        {/* Logo */}
        <img
          src={chameleonLogo}
          alt="Chameleon"
          className="h-14 md:h-16 w-auto"
        />

        {/* Heading */}
        <h1 className="mt-10 text-[64px] md:text-[96px] leading-[0.95] tracking-[-0.06em] font-extrabold font-jakarta max-w-6xl">

          <DynamicWeightText
            label="Deliver Unique"
            fromWeight={700}
            toWeight={900}
            strength={35}
            color="#17151d"
            fontSize="inherit"
          />

          <br />

          <DynamicWeightText
            label="Experiences at Scale"
            fromWeight={700}
            toWeight={900}
            strength={35}
            color="#5E43A5"
            fontSize="inherit"
          />

        </h1>

        {/* Description */}
        <p className="mt-10 max-w-3xl text-[22px] leading-relaxed text-[#666] font-inter">

          The Ultra Personalization Engine leverages real-time behavior and
          deep contextual ML to dynamically assemble interfaces for every
          individual user in milliseconds.

        </p>

        {/* CTA Buttons */}
        <div className="flex flex-wrap items-center justify-center gap-4 mt-14">

          {/* Explore Button */}
          <Link
            to="/home"
            className="px-8 py-4 rounded-full bg-[#17151d] text-white text-sm font-semibold shadow-lg hover:scale-105 transition-all duration-300 font-inter"
          >

            Explore

          </Link>

          {/* Secondary Button */}
          <button className="px-8 py-4 rounded-full border border-[#d7d3df] text-[#17151d] text-sm font-semibold hover:bg-white transition-colors duration-300 font-inter">

            Read the Docs

          </button>

        </div>

      </section>

    
  );
}