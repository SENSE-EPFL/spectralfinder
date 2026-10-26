// Typing in a table dropdown's search box (e.g. a gas name): Dash's table catches Backspace and
// Delete to clear the whole cell, and the arrow / Home / End keys to move to another cell, so a
// typo couldn't be corrected -- the box stayed stuck on the wrong text. Those keys now stay with
// the search box: stopped here, at the very start of the event's path (window, capture phase),
// before the table sees them. Their default action (deleting a character, moving the cursor) is
// left alone. Up / Down / Enter / Escape still reach the dropdown, to pick an option.
window.addEventListener("keydown", (event) => {
    const editingKeys = ["Backspace", "Delete", "ArrowLeft", "ArrowRight", "Home", "End"];
    const box = event.target;
    if (editingKeys.includes(event.key) && box instanceof HTMLInputElement
            && box.closest(".dash-spreadsheet .Select-input")) {
        event.stopPropagation();
    }
}, true);

// In the gas column's search box a zero is read as the letter O, so "N20" finds N2O: no
// molecule name has a zero (an isotopologue does, (70Ge)H4, so the isotope column is left
// alone). Done on the input event before React reads it, through the native value setter:
// assigning box.value directly would also update React's copy, and it would then see no change.
const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
window.addEventListener("input", (event) => {
    const box = event.target;
    if (box instanceof HTMLInputElement && box.value.includes("0")
            && box.closest(".dash-spreadsheet td[data-dash-column='molecule'] .Select-input")) {
        const cursor = box.selectionStart;
        setValue.call(box, box.value.replaceAll("0", "O"));
        box.setSelectionRange(cursor, cursor);
    }
}, true);
