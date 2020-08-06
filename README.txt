Default we have set the model to take the preloaded weights from pwf file and run the model based in the saved weights.

LSTM and BLSTM:

To run the pre trained model just keep the pwf file in the same folder as the code 
and run the model.

To train the model LSTM or BLSTM:

In BLSTM code "BLSTM.py" we have to comment the line number 109
then hit run as it is the line where it loads existing weights.


In LSTM code "LSTM.py" we have to comment line number 109 and hit run as it is the line where it loads existing weights.

The pwf files for LSTM.py is named as "dict_model.pwf"
The Pwf file for "BLSTM.py" is named as 
"dict_model.py"

To run with existing weights uncomment the 109 line as it is where the model weights are loaded and uncomment the break in for loop in main function which runs for 10 epochs.

Accuracy of LSTM model - 56%
Accuracy of BLSTM model - 58%
*Note  -  As we have named both the pwf files as "dict_model.pwf" keep both the codes at different folders with their respective pwf files. 
