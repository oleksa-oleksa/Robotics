#include <stdio.h>

int main()
{
	while (1) {
		//Get the current position
		current_position = read_current_position();

		//Calculate the error
		error = target_position - current_position;

		//Calculate the control variable
		control_variable = kp * error;

		// if CV is positive, run the motor clockwise and run forward
		// else run countercockwise and run backward
		if (control_variable > 0) {
			motor_pwm(control_variable);
		}
		else if (control_variable < 0) {
			motor_pwm(-control_variable);
		}
		else {
			motor_stop();
		}
	}
}