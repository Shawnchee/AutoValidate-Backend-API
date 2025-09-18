/**
 InputValidator Class
 
 Built-in Validators:
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

 Custom Validator Registration:
 InputValidator.registerValidator('validatorName', {
   allowedChars: /regex/,           // Allowed characters
   maxLength: number,               // Maximum length
   transform: (value) => value,     // Transform input (eg, toUpperCase)
   pattern: /regex/,                // Final validation pattern
   keypressError: "Error message",  // Error when invalid char entered
   validationError: "Error message", // Error when input invalid
   errorTimeout: 2000               // How long error message shows (ms)
 });

 Custom Validator Application:
 InputValidator.applyCustomValidator(inputElement, 'validatorName');
 */

class InputValidator {
    // Store custom validators
    // ---- IC Number ----
    static validateIC(input) {
        input.addEventListener("keypress", (e) => {
        if (!/[0-9]/.test(e.key)) {
            e.preventDefault();
            input.setCustomValidity("Only numbers are allowed for this field");
            input.reportValidity();
            setTimeout(() => {
                input.setCustomValidity("");
            }, 2000);
        }
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
        input.reportValidity(); 
        });
    }

    // ---- Car Plate ----
    static validateCarPlate(input) {
        input.addEventListener("keypress", (e) => {
            if (!/[A-Za-z0-9]/.test(e.key)) {
            e.preventDefault();
            input.setCustomValidity("Only letters and numbers are allowed for this field");
            input.reportValidity();
            setTimeout(() => {
                input.setCustomValidity("");
            }, 2000);
            }
        });

        input.addEventListener("input", () => {
            let value = input.value.toUpperCase();
            value = value.replace(/\s+/g, "");
            value = value.replace(/[^A-Z0-9]/g, "");
            if (value.length > 12) value = value.slice(0, 12);
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
            if (!/[0-9]/.test(e.key)) {
                e.preventDefault();
                input.setCustomValidity("Only numbers are allowed for this field");
                input.reportValidity();
                setTimeout(() => {
                    input.setCustomValidity("");
                }, 2000);
            }
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
    static customValidators = {};



    // Custom validators 
    // ---- Register Custom Validator ----
    static registerValidator(name, config) {
        this.customValidators[name] = config;
    }

    // ---- Apply Custom Validator ----
    static applyCustomValidator(input, validatorName) {
        const validator = this.customValidators[validatorName];
        if (!validator) {
            console.error(`Validator '${validatorName}' not found`);
            return;
        }

        // Apply keypress validation
        if (validator.allowedChars) {
            input.addEventListener("keypress", (e) => {
                if (!validator.allowedChars.test(e.key)) {
                    e.preventDefault();
                    input.setCustomValidity(validator.keypressError || "Invalid character");
                    input.reportValidity();
                    setTimeout(() => {
                        input.setCustomValidity("");
                    }, validator.errorTimeout || 2000);
                }
            });
        }

        // Apply input validation
        input.addEventListener("input", () => {
            let value = input.value;

            // Apply transformations
            if (validator.transform) {
                value = validator.transform(value);
            }

            // Apply max length
            if (validator.maxLength && value.length > validator.maxLength) {
                value = value.slice(0, validator.maxLength);
            }

            input.value = value;

            // Apply validation pattern
            if (validator.pattern) {
                if (validator.pattern.test(value)) {
                    input.setCustomValidity("");
                } else {
                    input.setCustomValidity(validator.validationError || "Invalid format");
                }
            }

            input.reportValidity();
        });
    }
    

}


export default InputValidator;
