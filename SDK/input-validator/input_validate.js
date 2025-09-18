/**
 InputValidator Class
 IC Number (NRIC)
  * - Digits only, max 12
  * - Auto-format to XXXXXX-XX-XXXX
 Car Plate
  * - Uppercase letters + digits
  * - No spaces, no symbols
  * - Must include at least one letter & number
 Postcode
  * - Digits only, max 5
  * - Must be exactly 5 digits
 */

class InputValidator {
    // ---- IC Number ----
    static validateIC(input) {
        input.addEventListener("keypress", (e) => {
        if (!/[0-9]/.test(e.key)) e.preventDefault(); 
        });

        input.addEventListener("input", () => {
        let value = input.value.replace(/\D/g, ""); 
        if (value.length > 12) value = value.slice(0, 12); 

        // Autoformat
        if (value.length > 6) value = value.slice(0, 6) + "-" + value.slice(6);
        if (value.length > 9) value = value.slice(0, 9) + "-" + value.slice(9);

        input.value = value;

        if (/^\d{6}-\d{2}-\d{4}$/.test(value)) {
            input.setCustomValidity("");
        } else {
            input.setCustomValidity("IC must follow XXXXXX-XX-XXXX format");
        }
        input.reportValidity(); // real-time popup
        });
    }

    // ---- Car Plate ----
    // ---- Car Plate ----
    static validateCarPlate(input) {
        // Block typing of invalid chars in real-time
        input.addEventListener("keypress", (e) => {
            if (!/[A-Za-z0-9]/.test(e.key)) {
            e.preventDefault();
            }
        });

        input.addEventListener("input", () => {
            let value = input.value.toUpperCase();       // force uppercase
            value = value.replace(/\s+/g, "");           // remove spaces
            value = value.replace(/[^A-Z0-9]/g, "");     // remove symbols
            if (value.length > 12) value = value.slice(0, 12); // increased limit for special plates
            input.value = value;

            const hasLetter = /[A-Z]/.test(value);
            const hasNumber = /\d/.test(value);

            if (!hasLetter || !hasNumber) {
            input.setCustomValidity("Car plate must contain at least 1 letter and 1 number");
            } else if (!/^([A-Z]{1,3}[0-9]{1,4}[A-Z]{0,2}|[A-Z]{4,12}[0-9]{1,4})$/.test(value)) {
            input.setCustomValidity("Invalid car plate format");
            } else {
            input.setCustomValidity("");
            }

            input.reportValidity(); 
        });
    }



    // ---- Postcode ----
    static validatePostcode(input) {
        input.addEventListener("keypress", (e) => {
            if (!/[0-9]/.test(e.key)) e.preventDefault(); 
        });

        input.addEventListener("input", () => {
            let value = input.value.replace(/\D/g, ""); 
            if (value.length > 5) value = value.slice(0, 5);
            input.value = value;

            if (/^\d{5}$/.test(value)) {
            input.setCustomValidity("");
            } else {
            input.setCustomValidity("Postcode must be 5 digits");
            }
            input.reportValidity();
        });
    }

}


export default InputValidator;
