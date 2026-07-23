import React from 'react';

import {
  Routes,
  Route,
  useParams,
} from 'react-router-dom';

import { ProductDetail } from './pages/productDetail';
import { LandingPage } from './pages/LandingPage';

import { PersonaShowcase } from './pages/Main';
import { PersonalizationProvider } from './context/personalizationContext';
// ...


// Product Detail Route Wrapper
function ProductDetailRoute() {

  const { productId } = useParams<{ productId: string }>();

  if (!productId) return null;

  return (
    <ProductDetail productId={decodeURIComponent(productId)} />
  );
}


// App Routes
export default function App() {

  return (
    <PersonalizationProvider>
      <Routes>

        <Route
          path="/"
          element={<LandingPage />}
        />


        {/* Homepage */}
        <Route
          path="/home"
          element={<PersonaShowcase  />}
        />

        {/* Product Detail */}
        <Route
          path="/product/:productId"
          element={<ProductDetailRoute />}
        />

        <Route path="/showcase" element={<PersonaShowcase />} />

      </Routes>
    </PersonalizationProvider>
  );
}