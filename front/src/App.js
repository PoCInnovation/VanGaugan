import React, {useState, useEffect} from 'react';

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
    const [imageNumber, setImageNumber] = useState(1);
    const [selectValue, setSelectValue] = useState(1);
    const [modelNames, setModelNames] = useState([]);
    const [selectedModel, setSelectedModel] = useState("celeba-30-e");
    const [hairLabel, setHairLabel] = useState('bold');
    const [genderLabel, setGenderLabel] = useState('male');

    useEffect(() => {
        fetch("/api/list-models")
            .then(res => res.json())
            .then(data => {
                setModelNames(data);
            })
            .catch(console.error)
    }, []);

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
                <FormControl style={{minWidth: 125}}>
                    <InputLabel id="input-label-1">Select a model</InputLabel>
                    <Select id="input-label-1" value={selectedModel} onChange={(ev) => setSelectedModel(ev.target.value)}>
                        {
                            modelNames.map((it) => {
                                return (
                                    <MenuItem key={it} value={it}>{it}</MenuItem>
                                )
                            })
                        }
                    </Select>
                </FormControl>
            </Grid>
            <Grid item>
                <img
                    src={`/api/${selectedModel}?image_number=${imageNumber}`}
                    alt="generator output">
                </img>
            </Grid>
            <Grid item>
                <FormControl style={{maxWidth: 125}}>
                    <TextField label="Image number" type="number" value={selectValue} onChange={handleChange}></TextField>
                </FormControl>
            </Grid>
            <Grid item>
                <h3>Labels selection</h3>
                <Grid container direction="row" spacing={5}>
                    <Grid item>
                        <FormControl style={{minWidth: 125}}>
                            <InputLabel id="input-label-2">Gender</InputLabel>
                            <Select id="input-label-2" value={genderLabel} onChange={(ev) => setGenderLabel(ev.target.value)}>
                                <MenuItem value={"None"}>None</MenuItem>
                                <MenuItem value={"male"}>male</MenuItem>
                                <MenuItem value={"female"}>female</MenuItem>
                            </Select>
                        </FormControl>
                    </Grid>
                    <Grid item>
                        <FormControl style={{minWidth: 125}}>
                            <InputLabel id="input-label-3">Hair</InputLabel>
                            <Select id="input-label-3" value={hairLabel} onChange={(ev) => setHairLabel(ev.target.value)}>
                                <MenuItem value={"None"}>None</MenuItem>
                                <MenuItem value={"bold"}>bold</MenuItem>
                                <MenuItem value={"blond"}>blond</MenuItem>
                                <MenuItem value={"brown"}>brown</MenuItem>
                            </Select>
                        </FormControl>
                    </Grid>
                </Grid>
            </Grid>
        </Grid>
    </div>
  );
}

export default App;
