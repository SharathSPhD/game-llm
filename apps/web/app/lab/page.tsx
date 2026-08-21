"use client";

import { useState } from "react";
import { EquilibriumLab } from "./components/EquilibriumLab";

export default function LabPage() {
  return (
    <div className="wrap">
      <section>
        <h1>Equilibrium Lab</h1>
        <p style={{ color: "var(--text-secondary)", maxWidth: "600px" }}>
          Run MMD, GDA, and QRE solvers on matrix games. Tune learning rate, magnetic strength,
          and step count to observe convergence behavior. Nash convergence is tracked along the
          trajectory; strategy evolution is shown on the simplex.
        </p>
      </section>

      <EquilibriumLab />
    </div>
  );
}
