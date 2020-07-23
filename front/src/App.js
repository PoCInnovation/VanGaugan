import React from 'react';

import {
    Grid,
    TextField,
    Select,
    MenuItem,
    InputLabel,
    FormControl
} from "@material-ui/core";

import './App.css';

function App() {
    const [label, setLabel] = React.useState('');
    const [imageNumber, setImageNumber] = React.useState(1);
    const [selectValue, setSelectValue] = React.useState(1);

    const handleChange = (event) => {
        event.preventDefault();
        let {value} = event.target;

        setSelectValue(value);

        if (value !== "" && value > 0 && value <= 64)
            setImageNumber(value)
    }

  return (
    <div className="App">
        <Grid container direction="column" justify="space-between" alignItems="center" spacing={5}>
            <Grid item>
                <h1>VanGaugan</h1>
            </Grid>
            <Grid item>
                <img
                    src={`/api/celeba?image_number=${imageNumber}&label=${label}`}
                    alt="generated image">
                </img>
            </Grid>
            <Grid item>
                <FormControl style={{maxWidth: 125}}>
                    <TextField label="Image number" type="number" value={selectValue} onChange={handleChange}></TextField>
                </FormControl>
            </Grid>
            <Grid item>
                <FormControl style={{minWidth: 125}}>
                    <InputLabel id="input-label">Label</InputLabel>
                    <Select id="input-label" value={label} onChange={(ev) => setLabel(ev.target.value)}>
                        <MenuItem value={"toto"}>toto</MenuItem>
                        <MenuItem value={"tata"}>tata</MenuItem>
                    </Select>
                </FormControl>
            </Grid>
        </Grid>
    </div>
  );
}

export default App;
