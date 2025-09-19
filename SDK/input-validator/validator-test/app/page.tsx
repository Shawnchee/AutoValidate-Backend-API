"use client";
import { useEffect } from "react";
import InputValidator from "input-validator";

export default function Home() {
  useEffect(() => {
    const icInput = document.getElementById("ic") as HTMLInputElement;
    const postcodeInput = document.getElementById("postcode") as HTMLInputElement;
    const carPlateInput = document.getElementById("carplate") as HTMLInputElement;

    if (icInput) {
      InputValidator.validateIC(icInput);
    }
    if (postcodeInput) {
      InputValidator.validatePostcode(postcodeInput);
    }
    if (carPlateInput) {
      InputValidator.validateCarPlate(carPlateInput);
    }
  }, []);

  return (
    <main style={{ padding: "2rem" }}>
      <h1>Validator SDK Test</h1>

      <div style={{ marginBottom: "1rem" }}>
        <label>IC Number:</label>
        <input id="ic" type="text" placeholder="YYMMDD-##-####" />
      </div>

      <div style={{ marginBottom: "1rem" }}>
        <label>Postcode:</label>
        <input id="postcode" type="text" placeholder="5 digits" />
      </div>

      <div style={{ marginBottom: "1rem" }}>
        <label>Car Plate:</label>
        <input id="carplate" type="text" placeholder="ABC1234" />
      </div>
    </main>
  );
}
